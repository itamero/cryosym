"""
Fourier shell correlation for cryosym, with EMalign's implementation (src.fsc).

EMalign and ASPIRE compute the same curve (the same shells, for odd and even box sizes). They differ in the resolution
read from it: EMalign interpolates the curve between shells and returns the frequency where it crosses the cutoff,
ASPIRE returns the frequency of the first shell at or below the cutoff, which is up to one shell coarser. cryosym uses
EMalign's resolution and plots everywhere.

    fsc_curve(vol_a, vol_b)                      the correlation of every shell
    fsc_resolution(vol_a, vol_b, cutoff)         the resolution in Angstrom at the cutoff
    plot_fsc(ref, vols, labels, cutoff, path)    EMalign's plot of one to three curves, saved to `path`

Volumes are aspire Volume objects (their pixel size is used) or 3D arrays (then give `pixel_size`), and must be aligned.
"""

from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
from aspire.volume import Volume
from src.fsc import FSCorr, fscres, plotFSC, plotFSC2, plotFSC3


def _array(vol):
    """The 3D float64 array of an aspire Volume (its first volume) or of an array."""
    return np.asarray(vol.asnumpy()[0] if isinstance(vol, Volume) else vol, dtype=np.float64)


def _pixel_size(vol, pixel_size):
    """`pixel_size` if given, else the pixel size of `vol` if it is a Volume that has one, else 1 (pixel units)."""
    if pixel_size is not None:
        return float(pixel_size)
    if isinstance(vol, Volume) and vol.pixel_size is not None:
        return float(vol.pixel_size)
    return 1.0


@contextmanager
def _no_show():
    """Disable plt.show, which EMalign's plot functions call, so that saving a plot never opens or blocks a window."""
    show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt.show = show


def fsc_curve(vol_a, vol_b):
    """FSC of two aligned volumes, one value per shell (EMalign's FSCorr)."""
    return FSCorr(_array(vol_a), _array(vol_b)).ravel()


def fsc_resolution(vol_a, vol_b, cutoff=0.5, pixel_size=None):
    """
    Resolution in Angstrom (or pixels) at which the FSC of two aligned volumes crosses `cutoff`, with the curve
    interpolated between shells (EMalign's fscres, the resolution printed by its plots).
    """
    fsc = fsc_curve(vol_a, vol_b)
    return 2 * _pixel_size(vol_a, pixel_size) * len(fsc) / fscres(fsc, cutoff)


def plot_fsc(ref, vols, labels=None, cutoff=0.5, pixel_size=None, path=None):
    """
    Plot the FSC of each of `vols` against `ref` with EMalign (plotFSC, plotFSC2 or plotFSC3) and save it to `path`.

    :param ref: Reference volume, e.g. the true map
    :param vols: One to three volumes aligned with `ref` (a single volume is also accepted)
    :param labels: Legend label of each volume (EMalign labels a single curve only in its title)
    :param cutoff: FSC cutoff of the resolution
    :param pixel_size: Pixel size in Angstrom, defaults to that of `ref`
    :param path: File to save the plot to (any matplotlib format); not saved if None
    :return: List of the resolutions in Angstrom, one per volume
    """
    vols = [vols] if isinstance(vols, (Volume, np.ndarray)) and _array(vols).ndim == 3 else list(vols)
    if not 1 <= len(vols) <= 3:
        raise ValueError(f"EMalign plots one to three FSC curves, got {len(vols)}")
    px = _pixel_size(ref, pixel_size)
    ref = _array(ref)
    with _no_show():
        if len(vols) == 1:
            res, fig = plotFSC(ref, _array(vols[0]), cutoff=cutoff, pixelsize=px)
            if path is not None:
                fig.savefig(path, dpi=300)
            resolutions = [res]
        else:
            plot = plotFSC2 if len(vols) == 2 else plotFSC3
            args = [v for vol in vols for v in (ref, _array(vol))]
            *resolutions, fig = plot(*args, labels=labels, cutoff=cutoff, pixelsize=px,
                                     figname=None if path is None else str(path))
    plt.close(fig)
    return [float(r) for r in resolutions]
