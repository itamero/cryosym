"""
Creates a cache file containing candidate rotation matrices and their common lines indices,
based on the symmetry type of the molecule. The cache is used to efficiently estimate relative
rotations between projection images.
"""
import math
import pickle
import logging

import numpy as np

from cryosym.group_elements import group_elements, scl_inds_by_sym, normalizer
from cryosym.gen_rotations_grid import gen_rotations_grid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def cryo_create_rotations_cache(
    sym, cache_file_name, cache_config
):
    """
    Creates a cache file containing:
     - All candidate rotation matrices: 'cache_rots' (n_cache_rots x 3 x 3)
     - Common lines indices: 'l_ij_ji_ind' ((n_cache_rots * n_cache_rots) x n_group_rots)
     - Self common lines indices: 'l_self_ind' (n_cache_rots x n_scl_pairs)
    n_group_rots is the order of the symmetry group, and n_scl_pairs is the number of self
    common lines indices used and n_cache_rots is the number of candidate rotations generated.

    Candidate rotations are generated from a filtered grid, based on the molecule's symmetry type.
    Common and self-common lines indices are computed between rotations and symmetry group elements.

    :param sym:                 Type of symmetry. eg: 'T', 'D5' or 'O'.
    :param cache_file_name:     Name of the generated cache file.
    :param cache_config:        cache_config = (rotation_resolution, n_theta,
                                                viewing_direction, in_plane_rotation)
        rotation_resolution: Number of samples per 2*pi (controls grid density).
        n_theta:             Angular resolution for common lines detection. Typically 360.
        in_plane_rotation:   Threshold for in-plane rotation, used in filtering.
        viewing_direction:   Threshold for viewing direction, used in filtering.


    :return:                    Name of generated cache file that includes
                                [cache_rots, l_self_ind, l_ij_ji_ind, resolution, n_theta].
    """
    (rotation_resolution, n_theta, viewing_direction, in_plane_rotation) = cache_config
    group_rots, scl_inds = group_elements(sym), scl_inds_by_sym(sym)
    logging.info("Creating candidate rotations set...")
    filtering_group_rots = group_elements(sym)
    cache_rots = candidate_rotations_set(
        filtering_group_rots, rotation_resolution, viewing_direction, in_plane_rotation
    )

    logging.info("Computing common lines and self common lines indices sets...")
    [l_self_ind, l_ij_ji_ind] = compute_cl_scl_indices(group_rots, scl_inds, n_theta, cache_rots)

    logging.info("Finished computing common lines and self common lines indices sets.")
    logging.info(
        "(%d x %d) common lines calculated and (%d x %d) self common lines calculated\n",
        len(l_ij_ji_ind), len(group_rots), len(l_self_ind), len(scl_inds)
    )

    with open(cache_file_name, "wb") as f:
        pickle.dump([cache_rots, l_self_ind, l_ij_ji_ind, rotation_resolution, n_theta], f)

    logging.info("Cache file created: %s\n", cache_file_name)
    return cache_file_name


def candidate_rotations_set(group_rots, resolution, viewing_direction, in_plane_rotation):
    """
    Candidate rotations are generated in an approximately equally spaced grid (based on the
    provided resolution) of rotations which is filtered based on the type of symmetry present
    in the molecule. The filtering is based on whether two rotations are in proximity after
    one is multiplied by some normalizer group element.

    :param group_rots:          Symmetry group elements: ((group order) x 3 x 3) numpy array
                                used for filtering the candidate rotations set.
    :param resolution:          Number of samples per 2pi  (see gen_rotations_grid for more details)
                                Default resolution is 150
    :param viewing_direction:   The viewing angle threshold
    :param in_plane_rotation:   The inplane rotation degree threshold

    :return:  numpy array (n_cache_rots x 3 x 3). Set of candidate rotation matrices after filtering
    """
    candidates_set = gen_rotations_grid(
        resolution
    )  # Generate approximately equally spaced rotations with specified resolution
    logging.info("With resolution %d, a set of %d rotations was originally generated.",resolution, len(candidates_set))
    close_idx = np.zeros(len(candidates_set), dtype=bool)

    # Precompute viewing directions (third column of each rotation matrix)
    viewing_dirs = candidates_set[:, :, 2]  # (n, 3)
    # Precompute group-applied viewing directions for filtering
    # all_g_applied_vds[g, j] = group_rots[g] @ viewing_dirs[j]
    all_g_applied_vds = (group_rots @ viewing_dirs.T).transpose(0, 2, 1)  # (|G|, n, 3)

    for r_i in range(len(candidates_set) - 1):
        if close_idx[r_i]:
            continue
        current = candidates_set[r_i]  # (3, 3)
        current_vd = current[:, 2]  # (3,)

        # Get remaining candidates (r_i+1 onwards, not yet marked close)
        remaining_mask = ~close_idx[r_i + 1:]
        remaining_indices = np.nonzero(remaining_mask)[0]  # indices relative to r_i+1
        if len(remaining_indices) == 0:
            continue
        remaining_indices_global = remaining_indices + r_i + 1

        # Viewing direction check: current_vd^T @ group_rots[g] @ remaining_vds[j]
        remaining_g_applied_vds = all_g_applied_vds[:, remaining_indices_global, :]  # (|G|, n_rem, 3)
        dots = remaining_g_applied_vds @ current_vd  # (|G|, n_rem)

        # Find (g, j) pairs where viewing direction is close
        g_pass, j_pass = np.nonzero(dots > viewing_direction)
        if len(g_pass) == 0:
            continue

        # For passing pairs, compute in-plane rotation check
        # transformed = group_rots[g] @ candidates[j_global]
        j_global = remaining_indices_global[j_pass]
        transformed = group_rots[g_pass] @ candidates_set[j_global]  # (n_pairs, 3, 3)
        # rot_diff = current^T @ transformed
        rot_diff = current.T @ transformed  # (n_pairs, 3, 3)
        # theta = |degrees(arctan(rot_diff[1,0] / rot_diff[0,0]))|
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = np.abs(np.degrees(np.arctan(rot_diff[:, 1, 0] / rot_diff[:, 0, 0])))

        passes = theta < in_plane_rotation
        if np.any(passes):
            close_global = np.unique(j_global[passes])
            close_idx[close_global] = True

    cache_rots = candidates_set[~close_idx]

    logging.info(
        "With resolution %d, after filtering, a set of %d rotations remain.\n",
        resolution, len(cache_rots)
    )
    return cache_rots


def compute_cl_scl_indices(group_rots, scl_inds, n_theta, cache_rots):
    """
    Computes the set of common lines induced by all rotation matrices pairs from cache_rots
    and the set of self common lines induced by each rotation matrix from cache_rots.

    :param group_rots:  Symmetry group elements: 3x3x(group order) numpy array
    :param scl_inds:    Indices of the symmetry group elements for which to compute the
                        self common lines.
    :param n_theta:     Angular resolution for common lines
    :param cache_rots:  Numpy array (n_cache_rots x 3 x 3) of rotation matrices

    :return:            l_self_ind      (n_cache_rots x (n_scl_pairs)) numpy array
                                        of self common lines indices.
                        l_ij_ji_ind     ((n_cache_rots x n_cache_rots) x n_group_rots)) numpy array
                                        of common lines indices.
                        Indices correspond to the common line induced by rotation matrices rot_i and
                        rot_j in range [0,1,...,n_theta-1]. Each pair of common line indices
                        calculated is stored as a single integer between 0 and (n_theta*n_theta)-1.
    """

    n_cache = len(cache_rots)
    n_group = len(group_rots)
    n_scl = len(scl_inds)

    pi = math.pi

    # Extract the columns we need from cache_rots
    # vd = viewing direction (3rd column), first2 = first two columns
    vd = cache_rots[:, :, 2]          # (n_cache, 3)
    first2 = cache_rots[:, :, :2]     # (n_cache, 3, 2)

    # === Self common lines (fully vectorized) ===
    scl_group = group_rots[scl_inds]  # (n_scl, 3, 3)

    # rel_rot = rot_i^T @ (scl_group[k] @ rot_i)
    # We only need rel_rot[0,2], rel_rot[1,2], rel_rot[2,0], rel_rot[2,1]
    # rel_rot[:, 2] = rot_i^T @ scl_group[k] @ rot_i[:, 2]  (for alpha_ij)
    # rel_rot[2, :] = rot_i[:, 2]^T @ scl_group[k] @ rot_i  (for alpha_ji)

    # scl_applied_vd[k,i] = scl_group[k] @ vd[i]  -> (n_scl, n_cache, 3)
    scl_applied_vd = (scl_group @ vd.T).transpose(0, 2, 1)
    # scl_applied_first2[k,i] = scl_group[k] @ first2[i]  -> (n_scl, n_cache, 3, 2)
    scl_applied_first2 = scl_group[:, None] @ first2[None, :]

    # ij_comp[k,i] = first2[i]^T @ scl_applied_vd[k,i]  -> (n_scl, n_cache, 2)
    # These are rel_rot[0,2] and rel_rot[1,2]
    ij_comp = (first2.transpose(0, 2, 1) @ scl_applied_vd[..., None]).squeeze(-1)
    # ji_comp[k,i] = vd[i]^T @ scl_applied_first2[k,i]  -> (n_scl, n_cache, 2)
    # These are rel_rot[2,0] and rel_rot[2,1]
    ji_comp = (vd[:, None, :] @ scl_applied_first2).squeeze(-2)

    # alpha_ij = arctan2(-rel_rot[0,2], rel_rot[1,2])
    alpha_ij = np.arctan2(-ij_comp[:, :, 0], ij_comp[:, :, 1]) + pi  # (n_scl, n_cache)
    # alpha_ji = arctan2(rel_rot[2,0], -rel_rot[2,1])
    alpha_ji = np.arctan2(ji_comp[:, :, 0], -ji_comp[:, :, 1]) + pi  # (n_scl, n_cache)

    l_self_ij = np.round(alpha_ij / (2 * pi) * n_theta).astype(int) % n_theta
    l_self_ji = np.round(alpha_ji / (2 * pi) * n_theta).astype(int) % n_theta

    # Pack and transpose to (n_cache, n_scl)
    l_self_ind = sub2ind([n_theta, n_theta], l_self_ij, l_self_ji)  # (n_scl, n_cache)
    l_self_ind = l_self_ind.T  # (n_cache, n_scl)

    # === Common lines (vectorized over all i, j) ===
    # Precompute group-applied quantities for all j
    # g_applied_vd[g,j] = group_rots[g] @ vd[j]  -> (n_group, n_cache, 3)
    g_applied_vd = (group_rots @ vd.T).transpose(0, 2, 1)
    # g_applied_first2[g,j] = group_rots[g] @ first2[j]  -> (n_group, n_cache, 3, 2)
    g_applied_first2 = group_rots[:, None] @ first2[None, :]

    # ij_comp[i,g,j] = first2[i]^T @ g_applied_vd[g,j]  -> (n_cache, n_group, n_cache, 2)
    ij_comp = (first2.transpose(0, 2, 1)[:, None, None] @ g_applied_vd[None, :, :, :, None]).squeeze(-1)
    # ji_comp[i,g,j] = vd[i]^T @ g_applied_first2[g,j]  -> (n_cache, n_group, n_cache, 2)
    ji_comp = (vd[:, None, None, None, :] @ g_applied_first2[None]).squeeze(-2)

    alpha_ij = np.arctan2(-ij_comp[:, :, :, 0], ij_comp[:, :, :, 1]) + pi
    alpha_ji = np.arctan2(ji_comp[:, :, :, 0], -ji_comp[:, :, :, 1]) + pi

    l_ij = np.round(alpha_ij / (2 * pi) * n_theta).astype(int) % n_theta
    l_ji = np.round(alpha_ji / (2 * pi) * n_theta).astype(int) % n_theta

    l_ij_ji_ind = sub2ind([n_theta, n_theta], l_ij, l_ji)  # (n_cache, n_group, n_cache)
    l_ij_ji_ind = l_ij_ji_ind.transpose(0, 2, 1).reshape(n_cache * n_cache, n_group)

    # Zero out diagonal entries (i == j)
    diag_rows = np.arange(n_cache) * n_cache + np.arange(n_cache)  # indices where i == j
    l_ij_ji_ind[diag_rows] = 0

    return l_self_ind.astype(int), l_ij_ji_ind


def compute_cl_indices(rot_i, rot_j, n_theta):
    """
    Compute the common line induced by rotation matrices rot_i and rot_j.

    :param rot_i:          3x3 numpy array rotation matrix.
    :param rot_j:          3x3 numpy array rotation matrix.
    :param n_theta:     Angular resolution for common lines detection.

    :return:        The indices of the common lines induced by rotations rot_i and rot_j.
                    Indices are integers in the range [0,1,...,L-1].
    """
    rel_rot = np.matmul(np.transpose(rot_i), rot_j)

    alpha_ij = np.arctan2(-rel_rot[0, 2], rel_rot[1, 2])
    alpha_ji = np.arctan2(rel_rot[2, 0], -rel_rot[2, 1])

    pi = math.pi
    alpha_ij += pi  # Shift from [-pi,pi] to [0,2*pi]
    alpha_ji += pi

    l_ij = (alpha_ij / (2 * pi)) * n_theta
    l_ji = (alpha_ji / (2 * pi)) * n_theta

    l_ij = int(np.round(l_ij) % n_theta)
    l_ji = int(np.round(l_ji) % n_theta)

    return l_ij, l_ji


def sub2ind(matrix_shape, row, col):
    """
    Converts (row, col) indices into 1D linear index. Traversing in row order.

    :param row: integer or array of integers between 0 and matrix_shape[0]-1.
    :param col: integer or array of integers between 0 and matrix_shape[1]-1.
    :param matrix_shape: Shape of the matrix (rows, cols)
    """
    _, cols = matrix_shape
    return cols * row + col


def ind2sub(matrix_shape, ind):
    """
    Converts 1D linear index into (row, col) indices. Traversing in row order.
    :param ind: integer or array of integers between 0 and matrix_shape[0]*matrix_shape[1]-1
    :param matrix_shape: Shape of the matrix (rows, cols)
    """
    rows, cols = matrix_shape
    row_indices = ind // cols  # Integer division to get row indices
    col_indices = ind % cols
    return row_indices, col_indices


def check_cache_file(cache_file_name, sym, rotation_resolution, n_theta):
    """
    Check provided cache file is not corrupt and matches the symmetry type.
    Note that n_theta is not checked (it is possible to change it later).

    :param cache_file_name: Name of the cache file that includes:
                            [cache_rots, l_self_ind, l_ij_ji_ind].
    :param sym:             Type of symmetry ('T' or 'O').
    """
    with open(cache_file_name, "rb") as f:
        cache_rots, l_self_ind, l_ij_ji_ind, resolution_file, n_theta_file = pickle.load(f)
    group_rots, scl_inds = group_elements(sym), scl_inds_by_sym(sym)
    if resolution_file != rotation_resolution:
        raise ValueError(
            f"Provided resolution {rotation_resolution} does not match cache "
            f"resolution {resolution_file}."
        )
    if n_theta_file != n_theta:
        raise ValueError(
            f"Provided n_theta {n_theta} does not match cache n_theta {n_theta_file}."
        )
    logging.info("Cache file contains %d candidate rotations.", len(cache_rots))
    logging.info(
        "Cache file contains (%d x %d) common lines and (%d x %d) self common lines\n",
        len(l_ij_ji_ind), len(group_rots), len(l_self_ind), len(scl_inds)
    )


if __name__ == "__main__":

    cryo_create_rotations_cache(
        sym = 'O',
        cache_file_name = "cache_example.pkl",
        cache_config = (40, 360, 0.996, 5),
    )
