"""
Experiments for the C_n algorithm of cryosym/cyclic.py on simulated projection images.

    python cn_experiment.py --sym C3 --n_imgs 20
    python cn_experiment.py --sym C4 --n_imgs 50 --fsc

Every run prints the diagnostics of each stage against the ground truth and appends one line to
data/cn_experiment/results.jsonl.
"""
import argparse
import json
import logging
import time

import numpy as np

from aspire.basis import DiracBasis3D
from aspire.noise import WhiteNoiseAdder
from aspire.reconstruction import MeanEstimator
from aspire.source import ArrayImageSource, Simulation
from aspire.utils.rotation import Rotation
from aspire.volume import Volume

from cryosym import cyclic
from cryosym.config import PROJECT_ROOT
from cryosym.volume_download.data_downloader import data_downloader

logger = logging.getLogger("cn_experiment")
OUT_DIR = PROJECT_ROOT / "data" / "cn_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def symmetry_check(vol, n):
    """Correlation between the volume and its rotation by 2 pi / n about e3 (1 for a C_n volume in standard position)."""
    v = vol.downsample(48)
    rot = Rotation(cyclic.rot_z(2 * np.pi / n)[None].astype(v.dtype))
    a, b = v.asnumpy().ravel(), v.rotate(rot).asnumpy().ravel()
    return float(np.corrcoef(a, b)[0, 1])


def simulate(sym, n_imgs, seed=0, snr=None, vol_size=None):
    vol = Volume.load(data_downloader(sym))
    if vol_size is not None and vol.resolution > vol_size:
        vol = vol.downsample(vol_size)
    noise = None if snr is None else WhiteNoiseAdder.from_snr(snr)
    src = Simulation(n=n_imgs, vols=vol, offsets=0, amplitudes=1, noise_adder=noise, seed=seed)
    return vol, src, np.array(src.rotations, dtype=float)


def stage_diagnostics(info, rots_gt, n):
    """Correctness of the pairwise estimates and of the vote against the ground truth."""
    grid, est, pairs = info["grid"], info["est"], info["pairs"]
    _, beta_gt, gamma_gt = cyclic.to_euler(rots_gt)
    # true folded orbit of every image
    fb = np.minimum(beta_gt, np.pi - beta_gt)
    fg = gamma_gt % np.pi

    def cell_dist(b_idx, g_idx, img):
        bi, gi = cyclic.fold(grid, b_idx, g_idx)
        db = np.degrees(np.abs(grid.beta[bi] - fb[img]))
        dg = np.degrees(np.abs((grid.gamma[gi] - fg[img] + np.pi / 2) % np.pi - np.pi / 2))
        return db, dg * np.sin(fb[img])  # gamma distance as an arc length on the sphere of third rows

    db_i, dg_i = cell_dist(est[:, 0], est[:, 1], pairs[:, 0])
    db_j, dg_j = cell_dist(est[:, 2], est[:, 3], pairs[:, 1])
    tol = 1.5 * grid.step
    ok = (np.hypot(db_i, dg_i) < tol) & (np.hypot(db_j, dg_j) < tol)
    vote_err = cyclic.orbit_error(info["beta"], info["gamma"], rots_gt)
    return {
        "pair_orbit_correct": float(ok.mean()),
        "vote_err_mean": float(vote_err.mean()),
        "vote_err_median": float(np.median(vote_err)),
        "vote_err_max": float(vote_err.max()),
        "vote_count_mean": float(info["count"].mean()),
    }, vote_err


def reconstruct(src, rots, sym, ds, maxiter=30):
    imgs = src.images[:].downsample(ds)
    rec_src = ArrayImageSource(imgs, angles=Rotation(rots.astype(np.float32)).angles, symmetry_group=sym)
    est = MeanEstimator(src=rec_src, basis=DiracBasis3D(rec_src.L, dtype=rec_src.dtype), maxiter=maxiter,
                        preconditioner="none")
    return est.estimate()


def run(sym, n_imgs=20, seed=0, snr=None, ds=70, step=5.0, n_theta=360, eta_p=22.0, fsc=False, ds_rec=120, n_jobs=-1,
        tag="", radial_power=0.0):
    n = int(sym[1:])
    vol, src, rots_gt = simulate(sym, n_imgs, seed, snr)
    sym_corr = symmetry_check(vol, n)
    logger.info(f"{sym}: volume {vol.resolution}^3, correlation with its rotation by 2pi/{n} about z: {sym_corr:.4f}")

    t0 = time.time()
    pf = cyclic.polar_ft(src.images[:].downsample(ds), n_theta=n_theta, radial_power=radial_power)
    rots, info = cyclic.estimate_rotations_cn(pf, n, step=step, eta_p=eta_p, n_jobs=n_jobs)
    t_est = time.time() - t0

    diag, vote_err = stage_diagnostics(info, rots_gt, n)
    rots_reg, err, glob = cyclic.register_cn(rots, rots_gt, n)
    res = {
        "sym": sym, "n_imgs": n_imgs, "seed": seed, "snr": snr, "ds": ds, "step": step, "eta_p": eta_p, "tag": tag,
        "radial_power": radial_power,
        "sym_corr": sym_corr, **diag,
        "ev_J": [float(x) for x in info["ev_J"]], "ev_F": [float(x) for x in info["ev_F"]],
        "ev_alpha": [float(x) for x in info["ev_alpha"]],
        "err_mean": float(err.mean()), "err_median": float(np.median(err)), "err_max": float(err.max()),
        "err_frac_below_5": float((err < 5).mean()), "time_est": t_est,
    }
    logger.info(f"per image error (deg): {np.round(err, 1)}")
    logger.info(f"per image vote error (deg): {np.round(vote_err, 1)}")
    if fsc:
        t0 = time.time()
        rec = reconstruct(src, rots_reg, sym, ds_rec)
        ref = vol.downsample(ds_rec)
        res_a, _ = rec.fsc(ref, cutoff=0.5, plot=str(OUT_DIR / f"fsc_{sym}_n{n_imgs}_seed{seed}{tag}.png"))
        res["fsc_05_angstrom"] = float(res_a[0])
        res["pixel_size"] = float(ref.pixel_size)
        res["time_rec"] = time.time() - t0
        rec.save(str(OUT_DIR / f"rec_{sym}_n{n_imgs}_seed{seed}{tag}.mrc"), overwrite=True)
    with open(OUT_DIR / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    logger.info("RESULT " + json.dumps(res))
    return res, rots, info, rots_gt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sym", default="C3")
    p.add_argument("--n_imgs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snr", type=float, default=None)
    p.add_argument("--ds", type=int, default=70)
    p.add_argument("--step", type=float, default=5.0)
    p.add_argument("--eta_p", type=float, default=22.0)
    p.add_argument("--fsc", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--radial_power", type=float, default=0.0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", force=True)
    run(**vars(args))
