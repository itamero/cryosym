from cryosym.group_elements import group_elements, coset_representatives
from cryosym.cryo_create_rotations_cache import compute_cl_indices
from cryosym.utils import multi_Jify
import pickle
import numpy as np
from scipy.linalg import eigh
from scipy import stats
import logging

def estimate_rotations(sym, cache_file_name, est_rel_rots, n_images, n_theta, pf):
    with open(cache_file_name, "rb") as f:
        R, _, _, _, _ = pickle.load(f)

    cache_selected_inds = np.zeros(n_images, dtype=int)
    cache_selected_count = np.zeros(n_images, dtype=int)
    for i in range(n_images):
        # Exclude the diagonal entries
        row = np.delete(est_rel_rots[i], i)
        mode_result = stats.mode(row, keepdims=False)
        cache_selected_inds[i] = mode_result.mode.astype(int)
        cache_selected_count[i] = int(mode_result.count)

    rots = np.zeros((n_images, 3, 3))
    for i in range(n_images):
        rots[i] = R[cache_selected_inds[i]]

    if sym == 'T' or (sym[0] == 'D' and sym not in ['D2']) or sym[0] == 'C':
        rots, coset_sync, J_sync = sync_Z2xZ2(sym, rots, n_theta, pf)
        coset_key = "cache_pi_x_indices" if sym[0] == 'C' else "cache_selected_r_indices"
        cache_selected_data = {
            "cache_selected_inds": cache_selected_inds,
            "cache_J_indices": J_sync,
            coset_key: coset_sync,
            "cache_selected_count": cache_selected_count,
        }
    elif sym == 'D2':
        rots, rho_sync, eps_sync, J_sync, coset_indices = sync_D3xZ2(sym, rots, n_theta, pf)
        cache_selected_data = {
            "cache_selected_inds": cache_selected_inds,
            "cache_J_indices": J_sync,
            "cache_selected_rho_indices": rho_sync,
            "cache_selected_eps_indices": eps_sync,
            "cache_coset_indices": coset_indices,
            "cache_selected_count": cache_selected_count,
        }
    elif sym == 'O' or sym == 'I':
        rots, J_sync = sync_Z2_J(sym, rots, n_theta, pf)
        cache_selected_data = {
            "cache_selected_inds": cache_selected_inds,
            "cache_J_indices": J_sync,
            "cache_selected_count": cache_selected_count,
        }
    else:
        raise NotImplementedError

    return rots, cache_selected_data


def sync_Z3(C, name):
    """
    Synchronize over Z₃ using spectral methods.

    C[i,j] contains the relative group element (0, 1, or 2)
    indicating that element_j = element_i * g^(C[i,j])

    We use the Hermitian matrix formulation with complex exponentials.
    """
    n = C.shape[0]
    omega = np.exp(2j * np.pi / 3)  # Primitive 3rd root of unity

    # Build Hermitian matrix H where H[i,j] = omega^(C[i,j])
    H = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            H[i, j] = omega ** C[i, j]

    # Compute top eigenvector
    eigenvalues, eigenvectors = eigh(H)
    v = eigenvectors[:, -1]  # Top eigenvector

    logging.info(f"{name} top 3 eigenvalues: {eigenvalues[-1:-4:-1]}")

    # Fix the gauge by setting element 0 as reference (identity)
    v = v / v[0]  # Now v[0] = 1

    # Extract group elements from phases
    # The phase of v[i] tells us the group element
    angles = np.angle(v)  # Phases in [-π, π]

    # Normalize angles to [0, 2π)
    angles = angles % (2 * np.pi)

    # Map angles to {0, 1, 2}
    # 0 → angle ≈ 0
    # 1 → angle ≈ 2π/3
    # 2 → angle ≈ 4π/3
    # Round to nearest multiple of 2π/3
    sync = np.round(angles * 3 / (2 * np.pi)) % 3
    sync = sync.astype(int)

    logging.info(f"{name} vector phases (degrees): {np.degrees(angles)}")
    logging.info(f"{name} synchronized values: {sync}")
    logging.info(f"{name} rho^1 indices: {np.where(sync == 1)[0].tolist()}")
    logging.info(f"{name} rho^2 indices: {np.where(sync == 2)[0].tolist()}")

    return sync


def sync_D3xZ2(sym, rots, n_theta, pf):
    """Synchronize over D₃ × Z₂ for D2 symmetry.

    Resolves three independent ambiguities per image:
    - ε (Z₂): whether to apply the epsilon coset representative
    - ρ (Z₃): which power of rho to apply (0, 1, or 2)
    - J (Z₂): whether to apply J-conjugation

    First solves the two Z₂ problems (J, ε), then rebuilds and solves
    the Z₃ problem (ρ) using the corrected rotations.
    """
    n_imgs = len(rots)
    coset_reps = coset_representatives('D2')
    rho = coset_reps[1]
    rho2 = coset_reps[2]
    eps = coset_reps[3]
    J = np.diag([1.0, 1.0, -1.0])

    def apply(g_idx, R):
        if g_idx < 6:
            return coset_reps[g_idx] @ R
        else:
            return J @ coset_reps[g_idx - 6] @ R @ J.T

    # Initialize matrices
    C_J = np.ones((n_imgs, n_imgs), dtype=int)
    C_eps = np.ones((n_imgs, n_imgs), dtype=int)
    C_rho = np.zeros((n_imgs, n_imgs), dtype=int)

    for i in range(n_imgs):
        C_rho[i, i] = 0
        for j in range(i + 1, n_imgs):
            Ri = rots[i]
            Rj = rots[j]

            scores = np.zeros(12)
            for g in range(12):
                Rj_g = apply(g, Rj)
                scores[g] = common_line_score(sym, Ri, Rj_g, pf[i], pf[j], n_theta)

            best = int(np.argmax(scores))

            # Decode
            label_J = 1 if best < 6 else -1
            coset_idx = best % 6

            # eps: indices 0-2 are normal (+1), 3-5 are flipped (-1)
            label_eps = 1 if coset_idx < 3 else -1

            # rho power: 0,3->0; 1,4->1; 2,5->2
            rho_power = coset_idx % 3

            C_J[i, j] = C_J[j, i] = label_J
            C_eps[i, j] = C_eps[j, i] = label_eps

            # Store the RAW measured relative rotation
            C_rho[i, j] = (-rho_power) % 3
            C_rho[j, i] = rho_power

    # 1. Solve Z2 problems FIRST
    sync_J = sync_Z2(C_J, "D2 Handedness")
    sync_eps = sync_Z2(C_eps, "D2 Epsilon")

    # after sync_J and sync_eps are computed:
    rots_frame_aligned = rots.copy()
    for i in range(n_imgs):
        if sync_J[i] < 0:
            rots_frame_aligned[i] = J @ rots_frame_aligned[i] @ J.T
        if sync_eps[i] < 0:
            rots_frame_aligned[i] = eps @ rots_frame_aligned[i]

    # Rebuild C_rho but only test the 3 plain rho coset reps (no J-conjugated versions)
    C_rho_new = np.zeros((n_imgs, n_imgs), dtype=int)
    for i in range(n_imgs):
        for j in range(i + 1, n_imgs):
            Ri = rots_frame_aligned[i]
            Rj = rots_frame_aligned[j]
            scores = np.zeros(3)
            for k in range(3):
                Rj_k = coset_reps[k] @ Rj  # only 0,1,2 : rho^0,rho^1,rho^2
                scores[k] = common_line_score(sym, Ri, Rj_k, pf[i], pf[j], n_theta)
            best = int(np.argmax(scores))
            C_rho_new[i, j] = (-best) % 3
            C_rho_new[j, i] = best

    # now sync_Z3(C_rho_new, ...)
    sync_rho = sync_Z3(C_rho_new, "D2 Rho")

    # Apply transformations and compute coset indices
    rots_out = rots.copy()
    coset_indices = np.zeros(n_imgs, dtype=int)

    for i in range(n_imgs):
        rho_power = sync_rho[i]
        eps_applied = (sync_eps[i] < 0)
        coset_indices[i] = rho_power + (3 if eps_applied else 0)

        if eps_applied:
            rots_out[i] = eps @ rots_out[i]

        if rho_power == 1:
            rots_out[i] = rho @ rots_out[i]
        elif rho_power == 2:
            rots_out[i] = rho2 @ rots_out[i]

        if sync_J[i] < 0:
            rots_out[i] = J @ rots_out[i] @ J.T

    return rots_out, sync_rho, sync_eps, sync_J, coset_indices

def sync_Z2_J(sym, rots, n_theta, pf):
    """Synchronize over Z₂ (handedness only) for O and I symmetries.

    Resolves J-conjugation ambiguity by comparing common line scores
    between original and J-conjugated rotations for each image pair.
    """

    n_imgs = len(rots)
    C = np.ones((n_imgs, n_imgs), dtype=int)

    for i in range(n_imgs):
        for j in range(i+1,n_imgs):
            same_coset_score = common_line_score(sym, rots[i], rots[j], pf[i], pf[j], n_theta)
            opp_coset_score = common_line_score(sym, rots[i], multi_Jify(rots[j]), pf[i], pf[j], n_theta)
            if same_coset_score > opp_coset_score:
                C[i,j] = C[j,i] = 1
            else:
                C[i,j] = C[j,i] = -1

    sync_J = sync_Z2(C, "Handedness")
    rots_out = rots.copy()
    for i in range(n_imgs):
        if sync_J[i] < 0:
            rots_out[i] = multi_Jify(rots_out[i])

    return rots_out, sync_J




def sync_Z2xZ2(sym, rots, n_theta, pf):
    """Synchronize over Z₂ × Z₂ (coset × handedness) for T, Dn (n>2).

    Resolves two independent Z₂ ambiguities per image:
    - Coset: whether the rotation needs multiplication by a coset representative
    - Handedness: whether the rotation needs J-conjugation (J = diag(1,1,-1))
    """
    n_imgs = len(rots)

    # pick a representative of the non-trivial coset
    r = coset_representatives(sym)[1]
    J = np.diag([1.0, 1.0, -1.0])

    def apply(g, R):
        if g == 0:  # I
            return R
        elif g == 1:  # r
            return r @ R
        elif g == 2:  # J
            return J @ R @ J.T
        else:  # rJ  (they commute)
            return r @ J @ R @ J.T


    C_coset = np.ones((n_imgs, n_imgs), dtype=int)  # +1 = same coset
    C_J = np.ones((n_imgs, n_imgs), dtype=int)  # +1 = same handedness

    for i in range(n_imgs):
        for j in range(i + 1, n_imgs):
            Ri = rots[i]
            Rj = rots[j]

            # evaluate the four possibilities
            scores = np.zeros(4)
            for g in range(4):
                Rj_g = apply(g, Rj)
                scores[g] = common_line_score(sym, Ri, Rj_g,
                                              pf[i], pf[j], n_theta)

            # winner → edge labels
            best = int(np.argmax(scores))
            # mapping:  0:I, 1:r, 2:J, 3:rJ
            label_coset = 1 if best in (0, 2) else -1  # 0,2 → same coset
            label_J = 1 if best in (0, 1) else -1  # 0,1 → same handedness

            C_coset[i, j] = C_coset[j, i] = label_coset
            C_J[i, j] = C_J[j, i] = label_J

    sync_coset = sync_Z2(C_coset, "Coset")
    sync_J = sync_Z2(C_J, "Handedness")

    rots_out = rots.copy()
    for i in range(n_imgs):
        if sync_coset[i] < 0:  # need to multiply by r
            rots_out[i] = r @ rots_out[i]
        if sync_J[i] < 0:  # need J-conjugation
            rots_out[i] = J @ rots_out[i] @ J.T

    return rots_out, sync_coset, sync_J


def sync_Z2(C, name):
    """Synchronize over Z₂ using spectral methods.

    Given a matrix C where C[i,j] = ±1 encodes pairwise relative labels,
    recovers globally consistent ±1 assignments via the top eigenvector.
    """
    eigenvalues, eigenvectors = eigh(C)
    v = eigenvectors[:, -1]
    logging.info(f"{name} top 3 eigenvalues: {eigenvalues[-1:-4:-1]}")
    sync = np.sign(v) + (v == 0)  # zero → +1
    sync = sync.astype(int)
    logging.info(f"{name} vector: {sync}")
    logging.info(f"{name} flip indices: {np.where(sync == -1)[0].tolist()}")
    return sync

def common_line_score(sym, rot1, rot2, p1, p2, n_theta):
    G = group_elements(sym)
    l_1 = np.zeros(len(G), dtype=int)
    l_2 = np.zeros(len(G), dtype=int)

    for k in range(len(G)):
        l_1[k], l_2[k] = compute_cl_indices(
            rot1, np.matmul(G[k], rot2), n_theta
        )
    Corrs_pi = np.real(np.matmul(p1.conj(), np.transpose(p2)))
    Corrs_cls = Corrs_pi[l_1, l_2]
    return np.prod(Corrs_cls)