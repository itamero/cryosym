import matplotlib.pyplot as plt
import shutil
import warnings
import mrcfile
import numpy as np
import pickle
import logging

from warnings import catch_warnings, filterwarnings
from aspire.utils.rotation import Rotation
from aspire.image import Image

from cryosym.config import read_mrc, align_volumes
from cryosym.group_elements import group_elements, coset_representatives

logger = logging.getLogger(__name__)


def load_projections_downsample(instack, ds_size, info=True):
    if info:
        logger.info(f"Loading mrc image stack file: {instack}")
    projs = Image.load(instack)

    if projs.shape[1] != projs.shape[2]:
        raise NotImplementedError(
            f"Only square projection images are supported."
            f"Provided projections have shape ({projs.shape[1]} x {projs.shape[2]})."
        )
    if info:
        logger.info(
            f"Loaded {projs.shape[0]} projections of size ({projs.shape[1]} x {projs.shape[2]})."
        )
        logger.info(
            f"Down-sampling projections to size ({ds_size} x {ds_size}).\n"
        )


    projs_ds = projs.downsample(ds_size)
    return projs_ds



def show(imgs, columns=5, figsize=(20, 10), colorbar=False, Title=None, save_path=None):
    """
    Plotting Utility Function.

    :param columns: Number of columns in a row of plots.
    :param figsize: Figure size in inches, consult `matplotlib.figure`.
    :param colorbar: Optionally plot colorbar to show scale.
        Defaults to True. Accepts `bool` or `dictionary`,
        where the dictionary is passed to `matplotlib.pyplot.colorbar`.
    """

    if imgs.stack_ndim > 1:
        raise NotImplementedError("`show` is currently limited to 1D image stacks.")

    # We never need more columns than images.
    columns = min(columns, imgs.n_images)
    rows = (imgs.n_images + columns - 1) // columns  # ceiling divide.

    # Create an empty colorbar options dictionary as needed.
    colorbar_opts = colorbar if isinstance(colorbar, dict) else dict()

    # Create a context manager for altering warnings
    with catch_warnings():
        # Filter off specific warning.
        # sphinx-gallery overrides to `agg` backend, but doesn't handle warning.
        filterwarnings(
            "ignore",
            category=UserWarning,
            message="Matplotlib is currently using agg, which is a"
            " non-GUI backend, so cannot show the figure.",
        )

        plt.figure(figsize=figsize)
        for i, im in enumerate(imgs.asnumpy()):
            plt.subplot(rows, columns, i + 1)
            plt.imshow(im, cmap="gray")
            if colorbar:
                plt.colorbar(**colorbar_opts)
        plt.suptitle(Title, fontsize=32)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

def angular_distance_rel_rots(rel_rot_est, rots_gt, sym, cache_file_name):
    ang_dist = np.inf
    # Each row i estimates R_i up to some g in G, h in normalizer representatives, J.
    with open(cache_file_name, "rb") as f:
        R, _, _, _, _ = pickle.load(f)

    minAngDists = np.zeros((len(rel_rot_est), len(rel_rot_est)))
    for row in range(len(rel_rot_est)):
        for col in range(len(rots_gt)):
            if row == col:
                continue
            dists = []
            for j in [0, 1]:
                for coset_rep_idx, coset_rep in enumerate(coset_representatives(sym)):
                    rot = R[rel_rot_est[row][col]]
                    rot_transformed = coset_rep @ rot
                    if j == 1:
                        rot_transformed = multi_Jify(rot_transformed)
                    dists.append(calculate_angular_distance_sym(rot_transformed, rots_gt[row], sym))
            minAngDists[row, col] = min(dists)
    return np.round(minAngDists, 2)

def mean_angular_distance_sym(rots, rots_gt, sym):
    # for each option of j, coset rep, register the rotations to the true rotations
    # and thus see which configuration is optimal to then check fsc for
    best_rots = None
    min_avg_dist = np.inf
    best_config = None

    for j in [0,1]:
        for coset_rep_idx, coset_rep in enumerate(coset_representatives(sym)):
            rots_transformed = coset_rep @ rots
            if j == 1:
                rots_transformed = multi_Jify(rots_transformed)
            dists = [calculate_angular_distance_sym(rots_transformed[i], rots_gt[i], sym)
                     for i in range(len(rots))]
            avg_dist = np.mean(dists)
            logger.info(f"j = {j}, coset_rep_idx = {coset_rep_idx}, avg_dist = {avg_dist}")

            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                best_rots = rots_transformed.copy()
                best_config = (j, coset_rep_idx)

    logger.info(f"Optimal Found: j={best_config[0]}, coset_idx={best_config[1]}, Min Avg Dist={min_avg_dist}")
    return best_rots


def calculate_angular_distance_sym(matrix1, matrix2, sym):
    """Calculate angular distance between two rotation matrices"""
    if matrix1 is None or matrix2 is None:
        return None

    if matrix1.shape != matrix2.shape:
        return None

    ang_dist = np.inf
    for g in group_elements(sym):
        ang_dist_g = Rotation.angle_dist(matrix1, g @ matrix2)
        if ang_dist_g < ang_dist:
            ang_dist = ang_dist_g
    return ang_dist * 180 / np.pi

def calculate_mse_distance_sym(matrix1, matrix2, sym):
    """Calculate MSE distance between two rotation matrices"""
    if matrix1 is None or matrix2 is None:
        return None

    if matrix1.shape != matrix2.shape:
        return None

    mse_dist = np.inf
    for g in group_elements(sym):
        diff = matrix1 - g @ matrix2
        mse_dist_g = np.mean(diff**2)
        if mse_dist_g < mse_dist:
            mse_dist = mse_dist_g
    return mse_dist


def min_norm_distance_sym(rots_1, rots_2, sym):
    """ "
    Find the mean angular distance between two sets of rotation matrices accounting for symmetry
    """
    assert len(rots_1) == len(rots_2) and rots_1.ndim == rots_2.ndim
    if rots_1.ndim == 2:
        rots_1 = np.expand_dims(rots_1, 0)
        rots_2 = np.expand_dims(rots_2, 0)
    norm_dists = np.zeros(len(rots_1))
    gR = group_elements(sym)
    for ind in range(len(rots_1)):
        R_sym = np.array([g_R @ rots_1[ind] for g_R in gR])
        diffs = R_sym - rots_2[ind]
        frobenius_norms = np.linalg.norm(diffs, axis=(1, 2))
        norm_dists[ind] = np.min(frobenius_norms)
    return np.mean(norm_dists)


def find_closest_rotation(true_rot, R, sym):
    dists = np.zeros(len(R))
    for i in range(len(R)):
        dists[i] = mean_angular_distance_sym(true_rot, R[i], sym)
    smallest_10 = np.sort(dists)[:10]
    print(f"smallest_10 = {smallest_10}")
    return np.argmin(dists)


def find_closest_rotation_norm(true_rot, R, sym):
    dists = np.zeros(len(R))
    for i in range(len(R)):
        dists[i] = min_norm_distance_sym(true_rot, R[i], sym)
    smallest_10 = np.sort(dists)[:10]
    print(f"smallest_10 = {smallest_10}")
    return np.argmin(dists)


# Based on EMalign main() but without parsing
def emalign_mrc_save(
    ref_mrc, query_mrc, output_mrc, n_projs=30, downsample=64, no_refine=False
):
    warnings.filterwarnings("ignore")

    vol1 = read_mrc(ref_mrc)
    vol2 = read_mrc(query_mrc)

    if vol1.ndim == 4 and vol1.shape[-1] == 1:
        vol1 = np.squeeze(vol1)
    elif vol1.ndim != 3:
        raise ValueError(
            "Volumes must be three-dimensional or fourdimensional with singleton first dimension "
        )

    # Handle the case where vol2 is 4D
    if (vol2.ndim == 4) and (vol2.shape[-1] == 1):
        vol2 = np.squeeze(vol2)
    elif vol2.ndim != 3:
        raise ValueError(
            "Volumes must be three-dimensional or fourdimensional with singleton first dimension "
        )

    if not (
        (vol1.shape[1] == vol1.shape[0])
        and (vol1.shape[2] == vol1.shape[0])
        and (vol1.shape[0] % 2 == 0)
    ):
        raise ValueError("All three dimensions of input volumes must be equal and even")

    if vol1.shape[0] != vol2.shape[0]:
        raise ValueError("Input volumes must be of same dimensions")

    class Struct:
        pass

    opt = Struct()
    opt.Nprojs = n_projs
    opt.downsample = downsample
    opt.no_refine = no_refine
    bestR, bestdx, reflect, vol2aligned, bestcorr = align_volumes(vol1, vol2, 1, opt)

    # Copy vol2 to save header
    shutil.copyfile(query_mrc, output_mrc)

    # Change and save
    mrc_fh = mrcfile.open(output_mrc, mode="r+")
    mrc_fh.set_data(vol2aligned.astype("float32").T)
    mrc_fh.set_volume()
    mrc_fh.update_header_stats()
    mrc_fh.close()

def multi_Jify(in_array):
    """
    Applies conjugation by J = diag(1,1,-1) to an array of 3x3 rotation matrices.
    """
    Jified_rot = np.array(in_array)
    Jified_rot[..., 2, :2] *= -1
    Jified_rot[..., :2, 2] *= -1
    return Jified_rot
