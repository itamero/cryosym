import logging
import numpy as np
import math
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from cryosym.group_elements import group_elements

RUN_DIAGNOSTICS = False


# Allowed candidate pairs of the pairwise maximization (see candidate_pairs_mask), angles in degrees
ETA = 1.0          # nearly parallel viewing directions modulo G
PLANE_BAND = 22.0  # band around the planes perpendicular to the axes of axes_U


def axes_U(sym):
    """
    Unit vectors u with g u = +-u for every g in G (one of each pair +-u), the axes of the coincident common lines
    lemma: e3 for C_n and D_n with n >= 3, e1, e2, e3 for D2, none for T, O and I.

    :param sym: Symmetry string
    :return: Array (n_axes, 3)
    """
    if sym == "C2":
        raise NotImplementedError("For C2 every equatorial line is invariant; the candidate set needs a direct test")
    if sym == "D2":
        return np.eye(3)
    if sym[0] in ("C", "D"):
        return np.array([[0.0, 0.0, 1.0]])
    return np.zeros((0, 3))


def candidate_pairs_mask(R, sym, eta=ETA, eta_a=4.8, eta_p=PLANE_BAND):
    """
    Boolean mask of the candidate pairs allowed in the pairwise maximization: mask[r, s] is True if
    1. |<v_r, g v_s>| < cos(eta) for every g in G (the induced common lines are defined),
    2. |<v_r, u>| < cos(eta_a) and |<v_s, u>| < cos(eta_a) for every u in axes_U(sym) (neither view is on an axis),
    3. max(|<v_r, u>|, |<v_s, u>|) > sin(eta_p) for every u in axes_U(sym) (the views are not both within eta_p of
       the plane perpendicular to u),
    where v_r, v_s are the viewing directions (third columns) of R[r], R[s].

    On the plane perpendicular to u all the induced common lines coincide, and near it they are too close to be told
    apart in noisy images, so the score of such a pair reduces to few distinct correlations and noise makes spurious
    maxima there. The band eta_p is therefore much wider than the grid spacing.

    :param R: Rotation cache, numpy array (r_candidates, 3, 3)
    :param sym: Symmetry string
    :param eta: Angle in degrees for nearly parallel viewing directions modulo G
    :param eta_a: Angle in degrees around the axes of axes_U(sym)
    :param eta_p: Angle in degrees around the planes perpendicular to them
    :return: Boolean mask (r_candidates, r_candidates)
    """
    views = R[:, :, 2]
    cos_eta = np.cos(np.radians(eta))
    mask = np.ones((len(R), len(R)), dtype=bool)
    for g in group_elements(sym):  # one group element at a time, to avoid an (r, r, |G|) array
        mask &= np.abs(views @ (g @ views.T)) < cos_eta
    n_defined = mask.sum()

    proj = np.abs(views @ axes_U(sym).T)  # (r_candidates, n_axes)
    on_axis = np.any(proj >= np.cos(np.radians(eta_a)), axis=1)
    mask &= ~on_axis[:, np.newaxis] & ~on_axis[np.newaxis, :]
    near_plane = proj <= np.sin(np.radians(eta_p))
    for u in range(proj.shape[1]):
        mask &= ~(near_plane[:, u][:, np.newaxis] & near_plane[:, u][np.newaxis, :])

    total_pairs = mask.size
    logging.info(
        f"Candidate pairs (eta = {eta} deg, eta_a = {eta_a:.2f} deg, eta_p = {eta_p:.2f} deg): "
        f"{mask.sum()} / {total_pairs} ({mask.sum() / total_pairs:.2%}); excluded "
        f"{1 - n_defined / total_pairs:.2%} as nearly parallel and {(n_defined - mask.sum()) / total_pairs:.2%} by the "
        f"axes {axes_U(sym).tolist()} ({on_axis.sum()} candidates on an axis)"
    )
    return mask


def estimate_relative_rotations(pf, cache_file_name, sym, true_rotations=None, eta=ETA, eta_a=None, eta_p=PLANE_BAND):
    """
    Estimate relative rotations between all pairs of projection images.

    For each pair (i, j), find cache rotation indices (Rij, Rji) that best explain
    the common lines between the polar Fourier transforms of images i and j.

    The maximization is over the candidate pairs of candidate_pairs_mask.

    :param pf: Polar Fourier transform of images, array (n_images, n_theta, n_rad).
    :param cache_file_name: Path to pickle file containing [R, l_self_ind, l_ij_ji_ind, resolution, _].
    :param sym: Symmetry string (e.g., 'C1', 'D7').
    :param true_rotations: Ground truth rotations (unused, reserved for diagnostics).
    :param eta: Angle in degrees for nearly parallel views (see candidate_pairs_mask).
    :param eta_a: Angle in degrees around the axes of axes_U, defaults to twice the grid spacing
                  360 / resolution of the cache.
    :param eta_p: Angle in degrees around the planes perpendicular to those axes.
    :return: Integer array (n_images, n_images) of cache rotation indices.
    """
    if RUN_DIAGNOSTICS:
        diagnostics = []
    else:
        diagnostics = None

    n_images, n_r = pf.shape[0], pf.shape[2]

    with open(cache_file_name, "rb") as f:
        R, l_self_ind, l_ij_ji_ind, resolution, _ = pickle.load(f)

    r_candidates = len(R)

    eta_a = 2 * 360 / resolution if eta_a is None else eta_a
    view_mask_global = candidate_pairs_mask(R, sym, eta, eta_a, eta_p)

    est_rel_rots = np.zeros((n_images, n_images))

    n_pairs = math.comb(n_images, 2)
    ii_inds = np.zeros(n_pairs).astype(int)
    jj_inds = np.zeros(n_pairs).astype(int)

    clmats = [0] * n_pairs
    ind = 0
    for ii in range(n_images):
        for jj in range(ii + 1, n_images):
            ii_inds[ind] = ii
            jj_inds[ind] = jj
            ind = ind + 1

    logging.info("Calculating self common lines scores")
    all_self_corrs = np.real(pf.conj() @ pf.transpose(0, 2, 1))
    all_extracted = all_self_corrs.reshape(n_images, -1)[:, l_self_ind]
    S_self = np.prod(np.maximum(0, all_extracted), axis=2)

    logging.info("Pre-computing all pairwise correlation matrices")
    pf_i_batch = pf[ii_inds]
    pf_j_batch = pf[jj_inds]
    all_Corrs_pi = np.real(pf_i_batch.conj() @ pf_j_batch.transpose(0, 2, 1))

    for ind in tqdm(range(n_pairs), desc="Calculating common lines scores", miniters=0):
        p_i = ii_inds[ind]
        p_j = jj_inds[ind]
        pair_diagnostics = diagnostics if ind < 3 else None
        clmats[ind] = max_correlation_pair_ind(
            all_Corrs_pi[ind], p_i, p_j, S_self, r_candidates, l_ij_ji_ind, sym,
            pair_diagnostics, R,
            view_mask_global=view_mask_global
        )
        if pair_diagnostics is not None:
            plot_top_correlations(diagnostics, pair_index=ind)

    for ind in range(n_pairs):
        p_i = ii_inds[ind]
        p_j = jj_inds[ind]
        est_rel_rots[p_i, p_j] = clmats[ind][0]
        est_rel_rots[p_j, p_i] = clmats[ind][1]

    return est_rel_rots.astype(int)


def max_correlation_pair_ind(
        Corrs_pi, p_i, p_j, S_self, r_candidates, l_ij_ji_ind, sym,
        diagnostics, R=None, view_mask_global=None
):
    """
    Find the rotation pair (ind1, ind2) maximizing the correlation score for images (p_i, p_j).

    :param Corrs_pi: Pre-computed correlation matrix np.real(pf_i.conj() @ pf_j.T), shape (n_theta, n_theta).
    When diagnostics is not None, also records top-10 correlations.
    """
    Sij = np.outer(S_self[p_i].conjugate(), S_self[p_j])
    np.fill_diagonal(Sij, 0)

    Corrs_cls = Corrs_pi.ravel()[l_ij_ji_ind]

    cl = np.prod(Corrs_cls, axis=1)
    c = cl.reshape(r_candidates, r_candidates)

    Sij = Sij * c

    if view_mask_global is not None:
        Sij[~view_mask_global] = -np.inf
    else:
        logging.warning(
            f"view_mask_global not provided for pair ({p_i}, {p_j}). "
            "Skipping view filtering."
        )

    ind1, ind2 = np.unravel_index(np.argmax(Sij), Sij.shape)

    if diagnostics is not None:
        corr_vals = Sij.flatten()
        sorted_idx = np.argsort(corr_vals)[::-1]
        top_k = min(10, len(sorted_idx))
        top_vals = corr_vals[sorted_idx[:top_k]]
        top_inds = np.array(np.unravel_index(sorted_idx[:top_k], Sij.shape)).T

        diagnostics.append({
            "pair": (int(p_i), int(p_j)),
            "best_val": Sij[ind1, ind2],
            "top_vals": top_vals,
            "top_inds": top_inds,
        })
    return ind1, ind2

def plot_top_correlations(diagnostics, pair_index=0):
    """Bar chart of top correlation values for a given image pair."""
    entry = diagnostics[pair_index]
    vals = entry["top_vals"]
    inds = entry["top_inds"]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(
        range(len(vals)),
        vals,
        tick_label=[str(tuple(int(x) for x in idx)) for idx in inds]
    )
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Top-{len(vals)} correlations for pair {entry['pair']}")
    plt.xlabel("Candidate rotation indices (cache)")
    plt.ylabel("Correlation value")

    valid_vals = vals[np.isfinite(vals)]
    if len(valid_vals) > 0:
        plt.ylim(0, max(1.0, np.max(valid_vals) * 1.1))
    else:
        plt.ylim(0, 1)

    for bar, val in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='black'
        )

    plt.tight_layout()
    plt.show()
