"""Print the runs recorded by cn_experiment.py in data/cn_experiment/results.jsonl as a table."""
import json
import sys

from cryosym.config import PROJECT_ROOT

COLS = ["sym", "n_imgs", "seed", "snr", "step", "radial_power", "tag", "pair_orbit_correct", "vote_err_mean", "err_mean", "err_median",
        "err_max", "err_frac_below_5", "fsc_05_angstrom", "pixel_size", "time_est"]

if __name__ == "__main__":
    last = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rows = [json.loads(line) for line in open(PROJECT_ROOT / "data" / "cn_experiment" / "results.jsonl")]
    rows = rows[-last:] if last else rows
    print(" | ".join(COLS))
    for r in rows:
        print(" | ".join(f"{r.get(c):.2f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in COLS))
