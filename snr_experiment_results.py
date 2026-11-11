"""
Ab-initio reconstruction from the simulated SNR stacks of snr_experiment.py, and a table of the results per group.

The estimates can only be compared with the truth up to the ambiguity set A = (N(G)/G) x Z2, whose elements act on a
rotation by a.R = J^d c R J^d (c a coset representative of N(G)/G, d the handedness bit), and up to G. Distances are
taken modulo G: d_G(X, Y) = min_g angle(X^T g Y). With a tolerance eta_v, the columns of the table are:

    pairs:          fraction of pairs (i, j) for which one a in A brings both pairwise estimates within eta_v of R_i
                    and R_j
    votes:          fraction of images whose voted rotation is within eta_v of R_i after some a in A
    median err:     median final error, after the global alignment over A done by cryo_abinitio_sym
    <5 deg:         fraction of images with final error below 5 degrees
    FSC:            resolution (Angstrom) at FSC = 0.5 between the reconstruction and the true volume

The grid floor printed with the table is the median distance from each true rotation to the nearest grid rotation
after the best a in A: the median error of a perfect run without refinement.

Usage: python snr_experiment_results.py D2 D3 T   (default: every group of snr_experiment.SYMS)

Every run writes to its own directory data/snr_experiment/runs/<date>_<time>/, with one subdirectory per group named
by the group and its EMDB entry, e.g. D3_emd14803/, holding <sym>_results.csv, <sym>_results.tex and
<sym>_snr_<label>_reconstruction.mrc. The run directory itself holds all_results.csv and all_results.tex with every
group of the run, config.json with the parameters and run.log. The simulated stacks are shared by all runs and stay
in data/snr_experiment/simulations/ (see snr_experiment.py).

Reproducibility: the orientations and the noise are seeded and the pipeline has no other randomness, so a rerun on
the same machine gives the same table. On another machine, the floating point libraries (FFTW, BLAS, FINUFFT) can
differ in the last bits; this is harmless where the best candidate wins clearly, but can flip near ties in the
pairwise argmax and the vote at low SNR, and so change those rows. config.json records the versions.
"""

import importlib.metadata
import inspect
import json
import logging
import pickle
import platform
import subprocess
import sys
from datetime import datetime

import mrcfile
import numpy as np
import pandas as pd

from cryosym.ab_initio_sym import cryo_abinitio_sym
from cryosym.config import SNR_EXPERIMENT_DIR
from cryosym.group_elements import coset_representatives, group_elements
from cryosym.volume_download.data_downloader import EMD_ID_MAP
from snr_experiment import SNRS, SYMS, load_volume, rotations_file, simulate_group, snr_file_tag, stack_file

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
    """Simulate and save the image stacks and true rotations of `sym` unless matching ones are already saved."""
    stack_files = [stack_file(sym, snr) for snr in snrs]
    if rotations_file(sym).exists() and all(f.exists() for f in stack_files):
        R_true = np.load(rotations_file(sym))
        with mrcfile.mmap(stack_files[0], permissive=True) as mrc:
            stack_size = mrc.data.shape[-1]
        if R_true.shape[0] == n_imgs and stack_size == L:
            return R_true, stack_files
    return simulate_group(sym, snrs, n_imgs, L, seed)


def group_dir(out_dir, sym):
    """Directory of the files of `sym` inside the run directory `out_dir`, named by the group and its EMDB entry."""
    return out_dir / f"{sym}_emd{EMD_ID_MAP[sym]}"


def snr_tex(snr):
    """SNR as written in the thesis table."""
    return r"$\infty$" if snr is None or pd.isna(snr) else f"{snr:g}"


def write_tex(dfs, path, floors):
    """
    Write the tables of one or more groups as a booktabs tabular in the style of the thesis results table.

    :param dfs: Dict from group to its results DataFrame
    :param path: Output .tex file
    :param floors: Dict from group to its grid floor in degrees
    """
    fmt = lambda x: "--" if pd.isna(x) else f"{x:.2f}"
    lines = [f"% grid floor of {sym} (median error of a perfect run without refinement): {floors[sym]:.2f} deg"
             for sym in dfs]
    lines += [
        r"\begin{tabular}{@{} l r r r r r r @{}}",
        r"\toprule",
        r"$G$ & SNR & pairs & votes & median err.\ [$^\circ$] & $<5^\circ$ & FSC [\AA] \\",
    ]
    for sym, df in dfs.items():
        lines.append(r"\midrule")
        group = f"${sym[0]}_{{{sym[1:]}}}$" if len(sym) > 1 else f"${sym}$"
        for k, row in df.reset_index(drop=True).iterrows():
            lines.append(
                f"{group if k == 0 else ''} & {snr_tex(row['snr'])} & {fmt(row['pairs'])} & {fmt(row['votes'])} & "
                f"{fmt(row['median_err'])} & {fmt(row['below_5'])} & {row['fsc']:.1f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def snr_experiment_results(
    sym,
    snrs=SNRS,
    n_imgs=40,
    L=129,
    seed=0,
    resolution="low",
    rotation_resolution=150,
    n_theta=360,
    ds_rot_est=79,
    ds_reconstruct=None,
    cg_max_iterations=30,
    eta_v=10.0,
    out_dir=SNR_EXPERIMENT_DIR,
):
    """
    Run cryo_abinitio_sym on the stack of every SNR level of `sym` and tabulate the results.

    :param sym: Symmetry group
    :param snrs: SNR levels, None for clean images
    :param n_imgs: Number of images per SNR level
    :param L: Size of the simulated images; saved stacks of another size or number of images are re-simulated
    :param seed: Seed of the orientations (used only if the stacks have to be simulated)
    :param resolution: Candidate grid preset, one of RESOLUTION_PRESETS
    :param rotation_resolution: Number of samples per 2*pi of the candidate grid
    :param n_theta: Number of rays per image
    :param ds_rot_est: Image size for rotation estimation
    :param ds_reconstruct: Image size for reconstruction, defaults to the size of the stacks
    :param cg_max_iterations: Maximum number of CG iterations of the reconstruction
    :param eta_v: Tolerance in degrees for a pairwise estimate or a vote to count as correct
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
        outvol = sym_dir / f"{sym}_snr_{snr_file_tag(snr)}_reconstruction.mrc"
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
            "fsc": float(np.atleast_1d(fsc_resolution)[0]),
        })
        logger.info(f"{sym}, SNR {snr_tex(snr)}: {rows[-1]}")

    df = pd.DataFrame(rows)
    floor = grid_floor(sym, cache_rots, R_true)
    df.insert(0, "group", sym)
    df["grid_floor"] = floor
    df.to_csv(sym_dir / f"{sym}_results.csv", index=False)
    write_tex({sym: df}, sym_dir / f"{sym}_results.tex", {sym: floor})
    logger.info(
        f"{sym} results (eta_v = {eta_v} deg, grid '{resolution}', grid floor {floor:.2f} deg):\n"
        f"{df.round(2).to_string(index=False)}"
    )
    return df


def write_combined_tables(syms, out_dir=SNR_EXPERIMENT_DIR):
    """Combine the saved per-group results of `syms` in the run directory `out_dir` into all_results.csv and .tex."""
    dfs = {sym: pd.read_csv(group_dir(out_dir, sym) / f"{sym}_results.csv") for sym in syms}
    floors = {sym: df["grid_floor"].iloc[0] for sym, df in dfs.items()}
    pd.concat(dfs.values()).to_csv(out_dir / "all_results.csv", index=False)
    write_tex(dfs, out_dir / "all_results.tex", floors)


if __name__ == "__main__":
    # Usage: python snr_experiment_results.py D2 D3 T   (default: every group of snr_experiment.SYMS)
    syms = sys.argv[1:] or SYMS

    run_dir = SNR_EXPERIMENT_DIR / "runs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
        "git_commit": commit,
        **{pkg: importlib.metadata.version(pkg) for pkg in ("numpy", "scipy", "aspire", "pyfftw")},
    }
    (run_dir / "config.json").write_text(json.dumps({"syms": syms, **params, "environment": environment}, indent=2))
    logger.info(f"Run directory: {run_dir}")

    for sym in syms:
        snr_experiment_results(sym, out_dir=run_dir)
        write_combined_tables(syms[: syms.index(sym) + 1], run_dir)
