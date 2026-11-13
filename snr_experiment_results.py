"""
Ab-initio reconstruction from the simulated SNR stacks of snr_experiment.py, and a table of the results per group.

The estimates can only be compared with the truth up to the ambiguity set A = (N(G)/G) x Z2, whose elements act on a
rotation by a.R = J^d c R J^d (c a coset representative of N(G)/G, d the handedness bit), and up to G. Distances are
taken modulo G: d_G(X, Y) = min_g angle(X^T g Y). With a tolerance eta_v, the columns of the table are:

    pairs:          fraction of pairs (i, j) for which one a in A brings both pairwise estimates within eta_v of R_i
                    and R_j
    votes:          fraction of images whose voted rotation is within eta_v of R_i after some a in A
    median err:     median final error, after the global alignment over A done by cryo_abinitio_sym
    <5 deg, <3 deg: fraction of images with final error below 5 and 3 degrees (<3 deg only in the csv and the log)
    FSC:            resolution (Angstrom) at FSC = 0.5 between the reconstruction and the true volume
    time:           wall-clock seconds of the cryo_abinitio_sym call, orientation estimation and reconstruction, from
                    the stored grid and common lines tables (building them is not included) to the saved volume; the
                    evaluation of the other columns is not included. With --jobs > 1 the groups share the CPU, so
                    the times are only comparable within runs with the same number of jobs (config.json records it)

The grid floor printed with the table is the median distance from each true rotation to the nearest grid rotation
after the best a in A: the median error of a perfect run without refinement (skipped with --no-grid-floor).

Usage: python snr_experiment_results.py [--n-imgs N] [--jobs J] [--no-grid-floor] [D2 D3 T ...]
    groups default to every group of snr_experiment.SYMS; N is the number of images per group and SNR level (default
    snr_experiment.N_IMGS), taken as the first N images of the pool of snr_experiment.N_POOL images, so that runs with
    different N are nested; J groups run in parallel processes (default 1).

Every run writes to its own directory data/snr_experiment/runs/n<N>_<date>_<time>/, with one subdirectory per group named
by the group and its EMDB entry, e.g. D3_emd14803/, holding <sym>_results.csv, <sym>_results.tex and, per SNR level,
<sym>_<level>_reconstruction.mrc, _rotations.npy (estimated rotations) and _errors.npy (error of each image). The run directory itself holds all_results.csv and all_results.tex with every
group of the run, config.json with the parameters and run.log. The simulated stacks are shared by all runs and stay
in data/snr_experiment/simulations/ (see snr_experiment.py).

Reproducibility: the orientations and the noise are seeded and the pipeline has no other randomness, so a rerun on
the same machine gives the same table. On another machine, the floating point libraries (FFTW, BLAS, FINUFFT) can
differ in the last bits; this is harmless where the best candidate wins clearly, but can flip near ties in the
pairwise argmax and the vote at low SNR, and so change those rows. config.json records the versions.
"""

import argparse
import importlib.metadata
import inspect
import json
import logging
import multiprocessing
import pickle
import platform
import os
import subprocess
import time
from datetime import datetime
from functools import partial

import mrcfile
import numpy as np
import pandas as pd

from cryosym.ab_initio_sym import cryo_abinitio_sym
from cryosym.config import SNR_EXPERIMENT_DIR
from cryosym.group_elements import coset_representatives, group_elements
from snr_experiment import (
    N_IMGS, N_POOL, SNRS, SYMS, group_name, group_seed, load_seeds, load_volume, noise_seed, rotations_file,
    simulate_group, snr_file_tag, stack_file, subset_stack,
)

logger = logging.getLogger(__name__)

J = np.diag([1.0, 1.0, -1.0])

# (in-plane rotation threshold, viewing direction threshold) of the candidate grid, as in ab_initio_simulation_demo
RESOLUTION_PRESETS = {"high": (2, 0.9985), "mid": (3, 0.9975), "low": (5, 0.996)}


def ambiguity_elements(sym):
    """Elements (f, d) of A: coset representative index f and handedness bit d."""
    return [(f, d) for d in (0, 1) for f in range(len(coset_representatives(sym)))]


def apply_ambiguity(sym, f, d, R):
    """a.R = J^d c_f R J^d for a stack of rotations R (..., 3, 3)."""
    R = coset_representatives(sym)[f] @ R
    return J @ R @ J if d else R


def dist_mod_G(X, Y, G):
    """Angle in degrees of X^T g Y, minimized over g in G, for broadcastable stacks X, Y (..., 3, 3)."""
    gY = np.einsum("gab,...bc->...gac", G, Y)
    traces = np.einsum("...ab,...gab->...g", X, gY)
    angles = np.arccos(np.clip((traces - 1) / 2, -1, 1))
    return np.degrees(angles.min(axis=-1))


def dists_over_ambiguity(sym, R_est, R_true):
    """d_G(a.R_est, R_true) for every a in A, stacked along the first axis."""
    G = group_elements(sym)
    return np.stack([dist_mod_G(apply_ambiguity(sym, f, d, R_est), R_true, G) for f, d in ambiguity_elements(sym)])


def pairs_correct_fraction(sym, cache_rots, est_rel_rots, R_true, eta_v):
    """Fraction of pairs i < j for which one a in A brings both R~_ij and R~_ji within eta_v of R_i and R_j."""
    n = len(R_true)
    i, j = np.triu_indices(n, k=1)
    d_ij = dists_over_ambiguity(sym, cache_rots[est_rel_rots[i, j]], R_true[i])
    d_ji = dists_over_ambiguity(sym, cache_rots[est_rel_rots[j, i]], R_true[j])
    correct = np.any(np.maximum(d_ij, d_ji) <= eta_v, axis=0)
    return correct.mean()


def votes_correct_fraction(sym, voted_rots, R_true, eta_v):
    """Fraction of images whose voted rotation is within eta_v of R_i after some a in A."""
    return np.mean(dists_over_ambiguity(sym, voted_rots, R_true).min(axis=0) <= eta_v)


def grid_floor(sym, cache_rots, R_true):
    """Median over images of the distance from R_i to the nearest grid rotation after the best a in A."""
    G = group_elements(sym)
    floors = []
    for R in R_true:
        d = [dist_mod_G(apply_ambiguity(sym, f, dd, cache_rots), R, G).min() for f, dd in ambiguity_elements(sym)]
        floors.append(min(d))
    return np.median(floors)


def ensure_stacks(sym, snrs, n_imgs, L, seed):
    """
    Stacks of the first `n_imgs` images of the pool of `sym` at every SNR level, and their true rotations.

    The pool (N_POOL images per level) is simulated unless a matching one is already saved: same size of images and
    the seeds of snr_experiment.group_seed and noise_seed, as recorded in the seeds.json of the group. If the orientations match
    but some SNR levels are missing or have other noise seeds, only those are simulated (the noise of a level
    depends only on the group and its SNR).
    """
    if n_imgs > N_POOL:
        raise ValueError(f"{n_imgs} images requested, but the pool has {N_POOL} (see snr_experiment.N_POOL)")
    pool_files = [stack_file(sym, snr) for snr in snrs]
    existing = [f for f in pool_files if f.exists()]
    recorded = load_seeds(sym)
    R_pool = None
    if (rotations_file(sym).exists() and existing and recorded is not None
            and recorded["orientations"] == group_seed(sym, seed)):
        saved = np.load(rotations_file(sym))
        with mrcfile.mmap(existing[0], permissive=True) as mrc:
            stack_shape = mrc.data.shape
        if saved.shape[0] == N_POOL and stack_shape == (N_POOL, L, L):
            R_pool = saved
            expected = {snr: None if snr is None else noise_seed(sym, snr, seed) for snr in snrs}
            stale = [snr for snr, f in zip(snrs, pool_files)
                     if not f.exists() or snr_file_tag(snr) not in recorded["noise"]
                     or recorded["noise"][snr_file_tag(snr)] != expected[snr]]
            if stale:
                simulate_group(sym, stale, L, seed, plot_snrs=snrs)
    if R_pool is None:
        R_pool, _ = simulate_group(sym, snrs, L, seed)
    return R_pool[:n_imgs], [subset_stack(sym, snr, n_imgs) for snr in snrs]


def group_dir(out_dir, sym):
    """Directory of the files of `sym` inside the run directory `out_dir`, named by the group and its EMDB entry."""
    return out_dir / group_name(sym)


def snr_tex(snr):
    """SNR as written in the thesis table."""
    return r"$\infty$" if snr is None or pd.isna(snr) else f"{snr:g}"


def group_tex(sym):
    """Group as written in the thesis, e.g. $D_{3}$, $\\mathbb{T}$."""
    return f"${sym[0]}_{{{sym[1:]}}}$" if len(sym) > 1 else f"$\\mathbb{{{sym}}}$"


def write_tex(dfs, path, floors):
    """
    Write the tables of one or more groups as a booktabs tabular in the style of the thesis results table.

    :param dfs: Dict from group to its results DataFrame
    :param path: Output .tex file
    :param floors: Dict from group to its grid floor in degrees (NaN or missing when not computed)
    """
    fmt = lambda x: "--" if pd.isna(x) else f"{x:.2f}"
    fmt_time = lambda x: "--" if pd.isna(x) else f"{x:.0f}"
    lines = [f"% grid floor of {sym} (median error of a perfect run without refinement): {floors[sym]:.2f} deg"
             for sym in dfs if not pd.isna(floors.get(sym, np.nan))]
    lines += [
        r"\begin{tabular}{@{} l r r r r r r r @{}}",
        r"\toprule",
        r"$G$ & SNR & pairs & votes & median err.\ [$^\circ$] & $<5^\circ$ & FSC [\AA] & time [s] \\",
    ]
    for sym, df in dfs.items():
        lines.append(r"\midrule")
        df = df.reset_index(drop=True)
        group = f"\\multirow{{{len(df)}}}{{*}}{{{group_tex(sym)}}}"  # centred over the rows of the group
        for k, row in df.iterrows():
            lines.append(
                f"{group if k == 0 else ''} & {snr_tex(row['snr'])} & {fmt(row['pairs'])} & {fmt(row['votes'])} & "
                f"{fmt(row['median_err'])} & {fmt(row['below_5'])} & {row['fsc']:.1f} & "
                f"{fmt_time(row.get('time', np.nan))} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def snr_experiment_results(
    sym,
    snrs=SNRS,
    n_imgs=N_IMGS,
    L=129,
    seed=0,
    resolution="low",
    rotation_resolution=150,
    n_theta=360,
    ds_rot_est=79,
    ds_reconstruct=None,
    cg_max_iterations=50,
    eta_v=10.0,
    compute_grid_floor=True,
    out_dir=SNR_EXPERIMENT_DIR,
):
    """
    Run cryo_abinitio_sym on the stack of every SNR level of `sym` and tabulate the results.

    :param sym: Symmetry group
    :param snrs: SNR levels, None for clean images
    :param n_imgs: Number of images per SNR level: the first n_imgs images of the pool (at most N_POOL)
    :param L: Size of the simulated images; a saved pool of another size is re-simulated
    :param seed: Base seed of the orientations and the noise (snr_experiment.group_seed; used only if the stacks have
                 to be simulated)
    :param resolution: Candidate grid preset, one of RESOLUTION_PRESETS
    :param rotation_resolution: Number of samples per 2*pi of the candidate grid
    :param n_theta: Number of rays per image
    :param ds_rot_est: Image size for rotation estimation
    :param ds_reconstruct: Image size for reconstruction, defaults to the size of the stacks
    :param cg_max_iterations: Maximum number of CG iterations of the reconstruction
    :param eta_v: Tolerance in degrees for a pairwise estimate or a vote to count as correct
    :param compute_grid_floor: Compute the grid floor (NaN in the table otherwise)
    :param out_dir: Run directory; the reconstructions and results tables go to group_dir(out_dir, sym)
    :return: DataFrame with one row per SNR level
    """
    in_plane_rotation, viewing_direction = RESOLUTION_PRESETS[resolution]
    sym_dir = group_dir(out_dir, sym)
    sym_dir.mkdir(parents=True, exist_ok=True)
    R_true, stack_files = ensure_stacks(sym, snrs, n_imgs, L, seed)
    true_vol = load_volume(sym, L)
    ds_reconstruct = ds_reconstruct or true_vol.resolution
    if true_vol.resolution != ds_reconstruct:
        true_vol = true_vol.downsample(ds_reconstruct)

    cache_rots = None
    rows = []
    for snr, stack_file in zip(snrs, stack_files):
        logger.info(f"{sym}, SNR {snr_tex(snr)}: reconstructing from {stack_file.name}")
        outvol = sym_dir / f"{sym}_{snr_file_tag(snr)}_reconstruction.mrc"
        start = time.perf_counter()
        est_vol, rots, details = cryo_abinitio_sym(
            sym,
            stack_file,
            outvol,
            n_theta=n_theta,
            rotation_resolution=rotation_resolution,
            viewing_direction=viewing_direction,
            in_plane_rotation=in_plane_rotation,
            n_r_perc=50,
            ds_rot_est=ds_rot_est,
            ds_reconstruct=ds_reconstruct,
            cg_max_iterations=cg_max_iterations,
            true_rotations=R_true,
            voxel_size=true_vol.pixel_size,
            return_details=True,
        )
        elapsed = time.perf_counter() - start  # the method only: the evaluation below is not timed
        if cache_rots is None:
            with open(details["cache_file_name"], "rb") as f:
                cache_rots = pickle.load(f)[0]

        # rots are already aligned to R_true over A by cryo_abinitio_sym
        errors = dist_mod_G(rots, R_true, group_elements(sym))
        fsc_resolution, _ = true_vol.astype(est_vol.dtype).fsc(est_vol, cutoff=0.5)

        rows.append({
            "snr": snr,
            "pairs": pairs_correct_fraction(sym, cache_rots, details["est_rel_rots"], R_true, eta_v),
            "votes": votes_correct_fraction(sym, cache_rots[details["cache_selected_inds"]], R_true, eta_v),
            "median_err": np.median(errors),
            "below_5": np.mean(errors < 5),
            "below_3": np.mean(errors < 3),
            "fsc": float(np.atleast_1d(fsc_resolution)[0]),
            "time": elapsed,
        })
        # Per image: the estimated rotations and their errors, for figures that need more than the table
        np.save(sym_dir / f"{sym}_{snr_file_tag(snr)}_rotations.npy", rots)
        np.save(sym_dir / f"{sym}_{snr_file_tag(snr)}_errors.npy", errors)
        logger.info(f"{sym}, SNR {snr_tex(snr)}: {rows[-1]}")

    df = pd.DataFrame(rows)
    floor = grid_floor(sym, cache_rots, R_true) if compute_grid_floor else np.nan
    df.insert(0, "group", sym)
    df["grid_floor"] = floor
    df.to_csv(sym_dir / f"{sym}_results.csv", index=False)
    write_tex({sym: df}, sym_dir / f"{sym}_results.tex", {sym: floor})
    logger.info(
        f"{sym} results ({n_imgs} images, eta_v = {eta_v} deg, grid '{resolution}', "
        f"grid floor {floor:.2f} deg):\n"
        f"{df.round(2).to_string(index=False)}"
    )
    return df


def write_combined_tables(syms, out_dir=SNR_EXPERIMENT_DIR):
    """Combine the saved per-group results of `syms` in the run directory `out_dir` into all_results.csv and .tex."""
    dfs = {sym: pd.read_csv(group_dir(out_dir, sym) / f"{sym}_results.csv") for sym in syms}
    floors = {sym: df["grid_floor"].iloc[0] if "grid_floor" in df else np.nan for sym, df in dfs.items()}
    pd.concat(dfs.values()).to_csv(out_dir / "all_results.csv", index=False)
    write_tex(dfs, out_dir / "all_results.tex", floors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ab-initio reconstruction of the SNR experiment stacks")
    parser.add_argument("syms", nargs="*", default=SYMS)
    parser.add_argument("--n-imgs", type=int, default=N_IMGS, help="images per group and SNR level")
    parser.add_argument("--jobs", type=int, default=1, help="groups run in parallel processes")
    parser.add_argument("--no-grid-floor", action="store_true", help="skip the grid floor")
    args = parser.parse_args()
    syms = args.syms

    run_dir = SNR_EXPERIMENT_DIR / "runs" / f"n{args.n_imgs:03d}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    run_dir.mkdir(parents=True)
    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    params = {
        name: p.default for name, p in inspect.signature(snr_experiment_results).parameters.items()
        if p.default is not inspect.Parameter.empty and name != "out_dir"
    }
    # Environment, since the low SNR rows can change with the floating point libraries (see the module docstring)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "git_commit": commit,
        **{pkg: importlib.metadata.version(pkg) for pkg in ("numpy", "scipy", "aspire", "pyfftw")},
    }
    params.update(n_imgs=args.n_imgs, n_pool=N_POOL, compute_grid_floor=not args.no_grid_floor, jobs=args.jobs)
    (run_dir / "config.json").write_text(json.dumps({"syms": syms, **params, "environment": environment}, indent=2))
    logger.info(f"Run directory: {run_dir}")

    kwargs = dict(n_imgs=args.n_imgs, compute_grid_floor=not args.no_grid_floor, out_dir=run_dir)
    if args.jobs > 1:
        # One process per group; the stacks of different groups are separate files, so they do not collide
        with multiprocessing.get_context("fork").Pool(min(args.jobs, len(syms))) as pool:
            pool.map(partial(snr_experiment_results, **kwargs), syms, chunksize=1)
        write_combined_tables(syms, run_dir)
    else:
        for sym in syms:
            snr_experiment_results(sym, **kwargs)
            write_combined_tables(syms[: syms.index(sym) + 1], run_dir)
