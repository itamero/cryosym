"""
Simulated projection images at the noise levels of the SNR experiment.

For every symmetry group, one ASPIRE `Simulation` is built per SNR from the group's EMD volume. All SNR levels of
a group share the same volume and the same orientations (same simulation seed), and differ only in the noise,
which is drawn from a seed that depends on the SNR. The SNR is ASPIRE's definition: the mean signal energy of the
clean images (inside the inscribed disk) divided by the noise variance.

Outputs (in data/snr_experiment/simulations/):
    - <sym>_snr_<label>.mrcs:   the full image stack per group and SNR
    - <sym>_true_rotations.npy: the true rotations, shared by all SNR levels of the group
    - <sym>_images.png:         one projection image, clean and at every finite SNR
    - all_groups_images.png:    the same image for every simulated group, one row per group
"""

import logging

import matplotlib.pyplot as plt
import mrcfile
import numpy as np

from aspire.noise import WhiteNoiseAdder
from aspire.source import Simulation
from aspire.volume import Volume

from cryosym.config import SNR_SIMULATIONS_DIR
from cryosym.volume_download.data_downloader import data_downloader

logger = logging.getLogger(__name__)

SYMS = ["D2", "D3", "D4", "D5", "D7", "T", "O", "I"]
SNRS = [None, 1, 1 / 2, 1 / 4]  # None is the clean (noise-free) level


def snr_label(snr):
    """Label of an SNR level for titles, e.g. 'clean', 'SNR = 1/2'."""
    if snr is None:
        return "clean"
    if snr >= 1:
        return f"SNR = {snr:g}"
    return f"SNR = 1/{round(1 / snr)}"


def snr_file_tag(snr):
    """Label of an SNR level for file names, e.g. 'clean', '1', '1_2'."""
    return snr_label(snr).replace("SNR = ", "").replace("/", "_")


def stack_file(sym, snr):
    """Path of the saved image stack of `sym` at `snr`."""
    return SNR_SIMULATIONS_DIR / f"{sym}_snr_{snr_file_tag(snr)}.mrcs"


def rotations_file(sym):
    """Path of the saved true rotations of `sym`."""
    return SNR_SIMULATIONS_DIR / f"{sym}_true_rotations.npy"


def load_volume(sym, L):
    """Load the EMD volume of `sym` (downloading it if needed), downsampled to L x L x L."""
    vol = Volume.load(data_downloader(sym), dtype=np.float64)
    if vol.resolution != L:
        vol = vol.downsample(L)
    return vol


def simulate_snr_levels(vol, n_imgs, snrs=SNRS, seed=0):
    """
    Build one Simulation per SNR level, with the same orientations and noise seeds that depend on the SNR.

    :param vol: Volume to project
    :param n_imgs: Number of images per simulation
    :param snrs: SNR levels, None for clean images
    :param seed: Seed of the orientations; the noise of the k-th SNR level uses seed + k
    :return: List of Simulation objects, one per SNR level
    """
    sims = []
    for k, snr in enumerate(snrs):
        noise_adder = None if snr is None else WhiteNoiseAdder.from_snr(snr, seed=seed + k)
        sim = Simulation(
            n=n_imgs,
            vols=vol,
            offsets=0,
            amplitudes=1,
            seed=seed,
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


def simulate_group(sym, snrs=SNRS, n_imgs=40, L=129, seed=0, img_idx=0):
    """
    Simulate `sym` at every SNR level, save the stacks and the true rotations, and plot <sym>_images.png.

    :return: True rotations (n_imgs, 3, 3) and the list of stack files, one per SNR level
    """
    logger.info(f"Simulating {sym} ({n_imgs} images of size {L})")
    sims = simulate_snr_levels(load_volume(sym, L), n_imgs, snrs, seed)
    np.save(rotations_file(sym), sims[0].rotations)
    for sim, snr in zip(sims, snrs):
        sim.images[:].save(stack_file(sym, snr), overwrite=True)
    plot_group_images(sym, snrs, img_idx)
    return sims[0].rotations, [stack_file(sym, snr) for snr in snrs]


def plot_snr_row(axes, sym, snrs=SNRS, img_idx=0):
    """Draw image `img_idx` of the saved stack of every SNR level of `sym` in a row of axes."""
    for ax, snr in zip(axes, snrs):
        with mrcfile.mmap(stack_file(sym, snr), permissive=True) as mrc:
            ax.imshow(np.array(mrc.data[img_idx]), cmap="gray")
        ax.set_title(snr_label(snr))
        ax.set_xticks([])
        ax.set_yticks([])


def plot_group_images(sym, snrs=SNRS, img_idx=0):
    """Plot image `img_idx` of `sym` at every SNR level into <sym>_images.png."""
    fig, axes = plt.subplots(1, len(snrs), figsize=(2.5 * len(snrs), 2.8))
    plot_snr_row(axes, sym, snrs, img_idx)
    fig.suptitle(f"{sym}: projection image {img_idx}")
    fig.tight_layout()
    fig.savefig(SNR_SIMULATIONS_DIR / f"{sym}_images.png", dpi=150)
    plt.close(fig)


def plot_all_groups(syms=SYMS, snrs=SNRS, img_idx=0):
    """Plot image `img_idx` of every group of `syms` with saved stacks into all_groups_images.png, one row each."""
    syms = [sym for sym in syms if all(stack_file(sym, snr).exists() for snr in snrs)]
    fig, axes = plt.subplots(len(syms), len(snrs), figsize=(2.5 * len(snrs), 2.5 * len(syms)), squeeze=False)
    for row, sym in enumerate(syms):
        plot_snr_row(axes[row], sym, snrs, img_idx)
        axes[row, 0].set_ylabel(sym, rotation=0, labelpad=20, fontsize=12)
    # Titles only on the top row
    for ax in axes[1:].ravel():
        ax.set_title("")
    fig.tight_layout()
    fig.savefig(SNR_SIMULATIONS_DIR / "all_groups_images.png", dpi=150)
    plt.close(fig)


def snr_experiment_images(syms=SYMS, snrs=SNRS, n_imgs=40, L=129, img_idx=0, seed=0):
    """
    Simulate every group at every SNR level, save the stacks and plot one projection image per level.

    :param syms: Symmetry groups
    :param snrs: SNR levels, None for clean images
    :param n_imgs: Number of images per group and SNR level
    :param L: Image size in pixels (the volume is downsampled to L x L x L before projecting)
    :param img_idx: Index of the image shown in the figures
    :param seed: Seed of the orientations
    """
    for sym in syms:
        simulate_group(sym, snrs, n_imgs, L, seed, img_idx)
    plot_all_groups(SYMS, snrs, img_idx)
    logger.info(f"Saved stacks and figures to {SNR_SIMULATIONS_DIR}")


if __name__ == "__main__":
    snr_experiment_images()
