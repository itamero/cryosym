import logging
import numpy as np
import math
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from cryosym.group_elements import group_elements

RUN_DIAGNOSTICS = False


def compute_admissible_pairs_mask(R, sym, threshold=1, equator_threshold=7.0):
    """
    Pre-compute a boolean mask over rotation pairs (R_i, R_j) from the cache.

    mask[i, j] is True if:
    1. For all group elements G_k: abs(<R_i[:, 2], G_k @ R_j[:, 2]>) < threshold
    2. R_i and R_j do not both lie near any common equatorial plane (D_n only)

    :param R: Rotation cache, numpy array (r_candidates, 3, 3)
    :param sym: Symmetry string (e.g., 'C1', 'D7')
    :param threshold: Max absolute inner product between viewing directions (default 1)
    :param equator_threshold: Angular threshold in degrees for equator proximity (default 7.0)
    :return: Boolean mask (r_candidates, r_candidates)
    """
    logging.info("Pre-computing viewing direction mask...")
    G = group_elements(sym)
    r_candidates = R.shape[0]

    all_views = R[:, :, 2]

    G_expanded = G[:, np.newaxis, :, :]
    R_expanded = R[np.newaxis, :, :, :]
    R_transformed = G_expanded @ R_expanded
    views_j_transformed = R_transformed[:, :, :, 2]

    # Shape: (r_candidates, r_candidates, n_group)
    abs_inner_prods = np.abs(np.einsum('id, gjd -> ijg', all_views, views_j_transformed))
    max_prods = np.max(abs_inner_prods, axis=2)
    view_mask = max_prods < threshold

    # Apply equator filtering for D_n symmetry
    if sym.startswith('D'):
        n = int(sym[1:])
        logging.info(f"Applying D_{n} equator filtering with threshold {equator_threshold}°...")

        eq_class = mark_all_equators_dn(all_views.T, equator_threshold, n)

        # For each pair (i, j), check if they share proximity to any equator
        shared_equators = eq_class[:, np.newaxis, :] & eq_class[np.newaxis, :, :]
        are_both_equators = np.any(shared_equators, axis=2)
        view_mask = view_mask & ~are_both_equators

        num_equator_filtered = np.sum(are_both_equators)
        logging.info(f"Filtered out {num_equator_filtered} pairs due to shared equator proximity")

    num_allowed = np.sum(view_mask)
    total_pairs = r_candidates * r_candidates
    logging.info(
        f"...viewing direction mask complete. "
        f"{num_allowed} / {total_pairs} pairs ({num_allowed / total_pairs:.2%}) "
        f"have distinct views."
    )
    return view_mask


def mark_all_equators_dn(sphere_grid, eq_filter_angle, n):
    """
    Mark viewing directions near equatorial planes for D_n symmetry.

    This is a Python translation of the MATLAB markAllEquatorsDn function.

    :param sphere_grid: Viewing directions, shape (3, nrot)
    :param eq_filter_angle: Angular threshold in degrees
    :param n: Degree of dihedral symmetry (D_n)
    :return: Boolean array (nrot, n+1) indicating proximity to each equator
    """
    nrot = sphere_grid.shape[1]
    angular_dists = np.zeros((nrot, n + 1))

    # Distance to primary equator
    proj_xy = sphere_grid.copy()
    proj_xy[2, :] = 0
    norms_xy = np.linalg.norm(proj_xy[:2, :], axis=0)
    norms_xy[norms_xy == 0] = 1  # Avoid division by zero
    proj_xy = proj_xy / norms_xy[np.newaxis, :]
    angular_dists[:, 0] = np.abs(np.sum(sphere_grid * proj_xy, axis=0))

    # Distances to n C2 rotation axes
    a = np.array([1.0, 0.0, 0.0])
    rotation_angle = 2 * np.pi / n

    for i in range(1, n + 1):
        dots = np.sum(sphere_grid * a[:, np.newaxis], axis=0)
        current_proj = sphere_grid - dots[np.newaxis, :] * a[:, np.newaxis]

        proj_norms = np.linalg.norm(current_proj, axis=0)
        proj_norms[proj_norms <= 1e-10] = 1
        current_proj = current_proj / proj_norms[np.newaxis, :]

        angular_dists[:, i] = np.abs(np.sum(sphere_grid * current_proj, axis=0))

        # Rotate axis for next iteration
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        a = np.array([
            cos_theta * a[0] - sin_theta * a[1],
            sin_theta * a[0] + cos_theta * a[1],
            a[2]
        ])

    eq_min_dist = np.cos(eq_filter_angle * np.pi / 180)
    eq_class = angular_dists > eq_min_dist

    return eq_class

def estimate_relative_rotations(pf, cache_file_name, sym, true_rotations=None):
    """
    Estimate relative rotations between all pairs of projection images.

    For each pair (i, j), find cache rotation indices (Rij, Rji) that best explain
    the common lines between the polar Fourier transforms of images i and j.

    :param pf: Polar Fourier transform of images, array (n_images, n_theta, n_rad).
    :param cache_file_name: Path to pickle file containing [R, l_self_ind, l_ij_ji_ind, _, _].
    :param sym: Symmetry string (e.g., 'C1', 'D7').
    :param true_rotations: Ground truth rotations (unused, reserved for diagnostics).
    :return: Integer array (n_images, n_images) of cache rotation indices.
    """
    if RUN_DIAGNOSTICS:
        diagnostics = []
    else:
        diagnostics = None

    n_images, n_r = pf.shape[0], pf.shape[2]

    with open(cache_file_name, "rb") as f:
        R, l_self_ind, l_ij_ji_ind, _, _ = pickle.load(f)

    r_candidates = len(R)

    view_mask_global = compute_admissible_pairs_mask(R, sym)

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
