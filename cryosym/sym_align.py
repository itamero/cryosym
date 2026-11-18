"""
Alignment of a symmetric reconstruction with a reference map of the same molecule, both in the standard orientation of
their symmetry group (cryosym.volume_download rotates the reference into it, and the reconstruction is built in it).

The two maps then differ by an element of the normalizer of the group, possibly composed with a reflection (the
handedness of an ab-initio reconstruction is arbitrary), and by a shift. A general alignment search (EMalign) can miss
this: for O it returned a wrong rotation of the mirrored map, 22 A at FSC 0.5 instead of 7 A. Here the candidates are
enumerated:

    every group     the 48 signed permutations of the axes (the octahedral group with reflections), which contain the
                    identity, a reflection, and the normalizer of T, O, I and D2 modulo the group
    D_n             also each of them composed with the rotation by pi / n about the symmetry axis (N(D_n) = D_2n)
    C_n             rotations about the symmetry axis on a grid of `cn_step` degrees over [0, 360 / n), each with and
                    without a reflection and a half turn about a perpendicular axis (N(C_n) = O(2))

and any extra candidates (e.g. the EMalign result) are added. Each candidate is shifted by the integer translation that
maximizes its cross-correlation with the reference (FFT), and the candidate with the largest normalized correlation is
returned.

Arrays are indexed (z, y, x) as ASPIRE and MRC store them, so the symmetry axis z of C_n and D_n is array axis 0.
"""

import itertools
import logging

import numpy as np
from scipy.ndimage import rotate

logger = logging.getLogger(__name__)


def _flip(vol, axis):
    """Reflect `vol` about the center of the grid along `axis` (index k -> L - k for even L, as ASPIRE centers at L//2)."""
    out = np.flip(vol, axis)
    return np.roll(out, 1, axis) if vol.shape[axis] % 2 == 0 else out


def signed_permutations(vol):
    """The 48 images of `vol` under the signed permutations of the axes, as (name, array)."""
    for perm in itertools.permutations(range(3)):
        moved = np.transpose(vol, perm)
        for signs in itertools.product((False, True), repeat=3):
            out = moved
            for ax, s in enumerate(signs):
                if s:
                    out = _flip(out, ax)
            yield f"perm {perm} flips {tuple(int(s) for s in signs)}", out


def _rotate_z(vol, degrees):
    """Rotate `vol` about the z axis (array axis 0) by `degrees`, linear interpolation."""
    return rotate(vol, degrees, axes=(1, 2), reshape=False, order=1) if degrees else vol


def candidates(vol, sym, cn_step=1.0):
    """The candidate alignments of `vol` for the group `sym`, as (name, array)."""
    if sym[0] == "C":
        n = int(sym[1:])
        for mirror, upside in itertools.product((False, True), repeat=2):
            base = _flip(vol, 2) if mirror else vol
            if upside:
                base = _flip(_flip(base, 0), 1)  # half turn about x
            for ang in np.arange(0, 360 / n, cn_step):
                yield f"mirror {int(mirror)} upside {int(upside)} angle {ang:g}", _rotate_z(base, ang)
        return
    yield from signed_permutations(vol)
    if sym[0] == "D":
        half = 180 / int(sym[1:])
        for name, out in signed_permutations(_rotate_z(vol, half)):
            yield f"rot_z {half:g} then {name}", out


def best_shift(vol, ref_fft, ref_norm):
    """`vol` shifted by the integer translation that maximizes its correlation with the reference, and that
    normalized correlation (the reference given by its FFT and norm; both maps are mean-subtracted first)."""
    v = vol - vol.mean()
    xc = np.real(np.fft.ifftn(np.fft.fftn(v) * ref_fft))
    idx = np.unravel_index(np.argmax(xc), xc.shape)
    shift = [-(k if k <= n // 2 else k - n) for k, n in zip(idx, xc.shape)]
    return np.roll(vol, shift, axis=(0, 1, 2)), xc[idx] / (ref_norm * np.linalg.norm(v)), shift


def align_sym(ref, vol, sym, extra=None, cn_step=1.0):
    """
    Align `vol` with `ref` (3D arrays on the same grid) over the candidates of `sym` and `extra` ({name: array}).

    :return: The aligned array and a dict with the name, normalized correlation and shift of the chosen candidate, and
             the correlation of each extra candidate
    """
    ref = np.asarray(ref, dtype=np.float64)
    r = ref - ref.mean()
    ref_fft, ref_norm = np.conj(np.fft.fftn(r)), np.linalg.norm(r)
    vol = np.asarray(vol, dtype=np.float64)
    best = (None, -np.inf, None, None)
    scores = {}
    pool = itertools.chain(candidates(vol, sym, cn_step), ((n, np.asarray(a, np.float64)) for n, a in (extra or {}).items()))
    for name, cand in pool:
        shifted, corr, shift = best_shift(cand, ref_fft, ref_norm)
        if extra and name in extra:
            scores[name] = corr
        if corr > best[1]:
            best = (name, corr, shift, shifted)
    name, corr, shift, aligned = best
    info = dict(candidate=name, correlation=float(corr), shift=[int(s) for s in shift],
                extra_correlations={k: float(v) for k, v in scores.items()})
    logger.info(f"Symmetry alignment ({sym}): {name}, shift {info['shift']}, correlation {corr:.4f}"
                + "".join(f"; {k}: {v:.4f}" for k, v in info["extra_correlations"].items()))
    return aligned, info
