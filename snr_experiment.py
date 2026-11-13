"""
Simulated projection images at the noise levels of the SNR experiment.

For every symmetry group, one ASPIRE `Simulation` of N_POOL images is built per SNR from the group's EMD volume. All
SNR levels of a group share the same volume and the same orientations, and differ only in the noise. This is the pool
of the group. An experiment with n_imgs <= N_POOL images uses the first n_imgs images of the pool (subset_stack), so
that experiments with different numbers of images are nested: the larger set contains the smaller one, with the same
orientations and the same noise.

Seeds: every group has its own orientations and its own noise. The orientation seed of a group is
group_seed(sym, seed) = seed + 1000 * (CRC32 of the group name mod 10^6), which depends only on the base seed and the
name, so it is the same in every run and on every machine, and adding or removing a group does not change the
others. The noise of an SNR level is drawn from group_seed + NOISE_SEED_OFFSETS[snr], so that adding or removing a
level does not change the noise of the others. The seeds of every saved stack are recorded in the seeds.json of the group, and
stacks whose recorded seeds differ from these (or that have no record) are simulated again.

The SNR is ASPIRE's definition: the mean signal energy of the clean images (inside the inscribed disk) divided by the
noise variance. The volume must be in the standard orientation of group_elements; load_volume checks this and refuses
a volume in another convention.

Outputs (in data/snr_experiment/simulations/), one directory per group named by the group and its EMDB entry:
    all_groups.png                 one projection image of every simulated group at every SNR level, one row each
    <sym>_emd<id>/
        seeds.json                 the orientation seed and the noise seed of every saved SNR level
        true_rotations.npy         the true rotations of the pool, shared by all SNR levels
        preview.png                one projection image, clean and at every finite SNR
        pool/<level>.mrcs          the pool stack of every level: clean.mrcs, snr_2.mrcs, snr_1.mrcs, snr_1_2.mrcs
        n<n_imgs>/<level>.mrcs     the first n_imgs images of the pool stacks, written when an experiment with
                                   n_imgs < N_POOL images needs them (e.g. n050/)
"""

import json
import logging
import zlib

import matplotlib.pyplot as plt
import mrcfile
import numpy as np

from aspire.noise import WhiteNoiseAdder
from aspire.source import Simulation
from aspire.volume import Volume

from cryosym.config import SNR_SIMULATIONS_DIR
from cryosym.volume_download.data_downloader import EMD_ID_MAP, data_downloader
from cryosym.volume_download.symmetry_group_conventions import (
    STANDARD_CONVENTIONS, SYMMETRY_CONVENTIONS, identify_symmetry_convention, sym_family,
)

logger = logging.getLogger(__name__)

SYMS = ["D2", "D3", "D4", "D5", "D7", "T", "O", "I"]
SNRS = [None, 2, 1, 1 / 2]  # None is the clean (noise-free) level
N_POOL = 200  # images simulated per group and SNR level
N_IMGS = 50  # default number of images of an experiment: the first N_IMGS images of the pool

# Noise seed of each SNR level is group_seed + offset (offsets below 1000, the spacing of the group seeds)
NOISE_SEED_OFFSETS = {1: 1, 1 / 2: 2, 1 / 4: 3, 2: 4}


def group_seed(sym, seed=0):
    """Orientation seed of `sym`: the base seed plus an offset that depends only on the name of the group."""
    return seed + 1000 * (zlib.crc32(sym.encode()) % 10**6)


def noise_seed(sym, snr, seed=0):
    """Seed of the noise of `sym` at the SNR level `snr`."""
    return group_seed(sym, seed) + NOISE_SEED_OFFSETS[snr]


def snr_label(snr):
    """Label of an SNR level for titles, e.g. 'clean', 'SNR = 1/2'."""
    if snr is None:
        return "clean"
    if snr >= 1:
        return f"SNR = {snr:g}"
    return f"SNR = 1/{round(1 / snr)}"


def snr_file_tag(snr):
    """Label of an SNR level for file names, e.g. 'clean', 'snr_1', 'snr_1_2'."""
    return snr_label(snr).replace("SNR = ", "snr_").replace("/", "_")


def group_name(sym):
    """Directory name of `sym`: the group and its EMDB entry, e.g. 'D3_emd14803'."""
    return f"{sym}_emd{EMD_ID_MAP[sym]}"


def group_dir(sym):
    """Directory of the simulated files of `sym`."""
    path = SNR_SIMULATIONS_DIR / group_name(sym)
    path.mkdir(parents=True, exist_ok=True)
    return path


def stack_file(sym, snr):
    """Path of the pool image stack of `sym` at `snr`."""
    path = group_dir(sym) / "pool" / f"{snr_file_tag(snr)}.mrcs"
    path.parent.mkdir(exist_ok=True)
    return path


def rotations_file(sym):
    """Path of the true rotations of the pool of `sym`."""
    return group_dir(sym) / "true_rotations.npy"


def seeds_file(sym):
    """Path of the record of the seeds of the pool stacks of `sym`."""
    return group_dir(sym) / "seeds.json"


def load_seeds(sym):
    """Recorded seeds of the pool stacks of `sym`: {"orientations": int, "noise": {snr_file_tag: int}}, or None."""
    path = seeds_file(sym)
    return json.loads(path.read_text()) if path.exists() else None


def subset_stack(sym, snr, n_imgs=N_IMGS):
    """
    Stack file with the first `n_imgs` images of the pool of `sym` at `snr`: the pool stack itself for N_POOL, a
    copy of its first images in the directory n<n_imgs>/ of the group otherwise. The copy is written again on every
    call, so that it always matches the pool.
    """
    if n_imgs > N_POOL:
        raise ValueError(f"{n_imgs} images requested, but the pool has {N_POOL} (see snr_experiment.N_POOL)")
    if n_imgs == N_POOL:
        return stack_file(sym, snr)
    path = group_dir(sym) / f"n{n_imgs:03d}" / stack_file(sym, snr).name
    path.parent.mkdir(exist_ok=True)
    with mrcfile.mmap(stack_file(sym, snr), permissive=True) as pool:
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(np.array(pool.data[:n_imgs]))
            mrc.set_image_stack()
            mrc.voxel_size = pool.voxel_size
    return path


def check_orientation(path, sym):
    """Raise if the volume at `path` is not in the standard orientation of `sym` (when conventions are defined)."""
    family = sym_family(sym)
    if family not in SYMMETRY_CONVENTIONS:
        return
    convention = identify_symmetry_convention(path, sym)
    if convention != STANDARD_CONVENTIONS[family]:
        raise ValueError(
            f"{path} is in the {convention} convention, not the standard {STANDARD_CONVENTIONS[family]}: the "
            f"simulated images would not have the symmetry axes of group_elements('{sym}')"
        )


def load_volume(sym, L):
    """Load the EMD volume of `sym` (downloading it if needed), check its orientation and downsample it to L x L x L."""
    path = data_downloader(sym)
    check_orientation(path, sym)
    vol = Volume.load(path, dtype=np.float64)
    if vol.resolution != L:
        vol = vol.downsample(L)
    return vol


def simulate_snr_levels(vol, sym, n_imgs, snrs=SNRS, seed=0):
    """
    Build one Simulation per SNR level of `sym`, with the orientations of group_seed(sym, seed) and the noise of
    noise_seed(sym, snr, seed).

    :param vol: Volume to project
    :param sym: Symmetry group (selects the seeds)
    :param n_imgs: Number of images per simulation
    :param snrs: SNR levels, None for clean images
    :param seed: Base seed
    :return: List of Simulation objects, one per SNR level
    """
    sims = []
    for snr in snrs:
        noise_adder = None if snr is None else WhiteNoiseAdder.from_snr(snr, seed=noise_seed(sym, snr, seed))
        sim = Simulation(
            n=n_imgs,
            vols=vol,
            offsets=0,
            amplitudes=1,
            seed=group_seed(sym, seed),
            noise_adder=noise_adder,
            dtype=np.float64,
        )
        if snr is not None:
            logger.info(f"{snr_label(snr)}: noise variance {noise_adder.noise_var:.4g}")
        sims.append(sim)

    # Same seed, so the same orientations at every SNR
    for sim in sims[1:]:
        assert np.allclose(sim.rotations, sims[0].rotations)
    return sims


def simulate_group(sym, snrs=SNRS, L=129, seed=0, img_idx=0, plot_snrs=None):
    """
    Simulate the pool of `sym` at the SNR levels `snrs`, save the stacks and the true rotations, and plot
    preview.png with the levels `plot_snrs` (default `snrs`).

    :return: True rotations (N_POOL, 3, 3) and the list of pool stack files, one per SNR level
    """
    logger.info(f"Simulating {sym} at {[snr_label(snr) for snr in snrs]} ({N_POOL} images of size {L}, "
                f"orientation seed {group_seed(sym, seed)})")
    sims = simulate_snr_levels(load_volume(sym, L), sym, N_POOL, snrs, seed)
    recorded = load_seeds(sym)
    if recorded is not None and recorded["orientations"] == group_seed(sym, seed):
        # Same orientations as the saved stacks: only some levels are (re)simulated, the others are kept
        saved = np.load(rotations_file(sym))
        if saved.shape == sims[0].rotations.shape and not np.allclose(saved, sims[0].rotations):
            raise ValueError(f"The rotations of {sym} differ from the saved ones of the same seed")
        noise = recorded["noise"]
    else:
        noise = {}  # new orientations: the saved stacks of other levels, if any, are stale
    np.save(rotations_file(sym), sims[0].rotations)
    for sim, snr in zip(sims, snrs):
        sim.images[:].save(stack_file(sym, snr), overwrite=True)
        noise[snr_file_tag(snr)] = None if snr is None else noise_seed(sym, snr, seed)
    seeds_file(sym).write_text(json.dumps({"orientations": group_seed(sym, seed), "noise": noise}, indent=2))
    plot_group_images(sym, plot_snrs or snrs, img_idx)
    return sims[0].rotations, [stack_file(sym, snr) for snr in snrs]


def plot_snr_row(axes, sym, snrs=SNRS, img_idx=0):
    """Draw image `img_idx` of the pool stack of every SNR level of `sym` in a row of axes."""
    for ax, snr in zip(axes, snrs):
        with mrcfile.mmap(stack_file(sym, snr), permissive=True) as mrc:
            ax.imshow(np.array(mrc.data[img_idx]), cmap="gray")
        ax.set_title(snr_label(snr))
        ax.set_xticks([])
        ax.set_yticks([])


def plot_group_images(sym, snrs=SNRS, img_idx=0):
    """Plot image `img_idx` of `sym` at every SNR level into the preview.png of the group."""
    fig, axes = plt.subplots(1, len(snrs), figsize=(2.5 * len(snrs), 2.8), squeeze=False)
    plot_snr_row(axes[0], sym, snrs, img_idx)
    fig.suptitle(f"{sym}: projection image {img_idx}")
    fig.tight_layout()
    fig.savefig(group_dir(sym) / "preview.png", dpi=150)
    plt.close(fig)


def plot_all_groups(syms=SYMS, snrs=SNRS, img_idx=0):
    """Plot image `img_idx` of every group of `syms` with saved stacks into all_groups.png, one row each."""
    syms = [sym for sym in syms if all(stack_file(sym, snr).exists() for snr in snrs)]
    fig, axes = plt.subplots(len(syms), len(snrs), figsize=(2.5 * len(snrs), 2.5 * len(syms)), squeeze=False)
    for row, sym in enumerate(syms):
        plot_snr_row(axes[row], sym, snrs, img_idx)
        axes[row, 0].set_ylabel(sym, rotation=0, labelpad=20, fontsize=12)
    # Titles only on the top row
    for ax in axes[1:].ravel():
        ax.set_title("")
    fig.tight_layout()
    fig.savefig(SNR_SIMULATIONS_DIR / "all_groups.png", dpi=150)
    plt.close(fig)


def snr_experiment_images(syms=SYMS, snrs=SNRS, L=129, img_idx=0, seed=0):
    """
    Simulate the pool of every group at every SNR level, save the stacks and plot one projection image per level.

    :param syms: Symmetry groups
    :param snrs: SNR levels, None for clean images
    :param L: Image size in pixels (the volume is downsampled to L x L x L before projecting)
    :param img_idx: Index of the image shown in the figures
    :param seed: Base seed (see group_seed)
    """
    for sym in syms:
        simulate_group(sym, snrs, L, seed, img_idx)
    plot_all_groups(SYMS, snrs, img_idx)
    logger.info(f"Saved stacks and figures to {SNR_SIMULATIONS_DIR}")


if __name__ == "__main__":
    snr_experiment_images()
