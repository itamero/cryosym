import math
import os
import logging

import numpy as np
import pandas as pd
import mrcfile

from aspire.operators import PolarFT
from aspire.utils.rotation import Rotation
from aspire.source import ArrayImageSource
from aspire.reconstruction import MeanEstimator
from aspire.basis import FFBBasis3D, DiracBasis3D

from cryosym.group_elements import supported_symmetry_group
from cryosym.cryo_create_rotations_cache import cryo_create_rotations_cache, check_cache_file
from cryosym.estimate_relative_rotations import estimate_relative_rotations
from cryosym.utils import mean_angular_distance_sym, load_projections_downsample, angular_distance_rel_rots
from cryosym.projection_guis.projection_gui_simulation import create_projection_viewer_simulation
from cryosym.projection_guis.projection_gui_class_avgs import create_projection_viewer_class_avgs
from cryosym.config import ROTATIONS_CACHE_DIR
from cryosym.estimate_rotations import estimate_rotations

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def cryo_abinitio_sym(
    sym,
    instack,
    outvol,
    n_theta=360,
    rotation_resolution=150,
    n_r_perc=50,
    viewing_direction=0.9985,
    in_plane_rotation=2,
    cg_max_iterations=50,
    ds_rot_est=70,
    ds_reconstruct=120,
    basis="Dirac",
    true_rotations=None,
    voxel_size=None,
    gui = False,
):
    """
    Ab-initio reconstruction of a symmetric molecule


    :param sym:                 Symmetry type. One of 'Cn','Dn','T','O','I' where n>1
    :param instack:             Name of MRC file containing the projections from which to estimate an ab-initio model.
    :param outvol:              Name (or path) of MRC file into which to save the reconstructed volume.
    :param rotation_resolution: (Optional) Number of samples per 2*pi  (see gen_rotations_grid for more details)
    :param in_plane_rotation:   (Optional) In-plane rotation angle threshold, used to determine proximity of rotations
                                for filtering i.e. creating "SO_G(3)"
    :param viewing_direction:   (Optional) Viewing direction angle threshold, used to determine proximity of rotations
                                for filtering i.e. creating "SO_G(3)"
    :param n_theta:             (Optional) Angular resolution for common lines detection.
                                Default 360.
    :param n_r_perc:            (Optional) Radial resolution for common line detection as a percentage of image size.
                                Default is half the width of the images.
    :param basis:               (Optional) Basis for volume reconstruction. Either "FFB" (Fourier-Bessel) or
                                "Dirac". Default "Dirac".
    :param cg_max_iterations:   (Optional) Maximum number of iterations for CG in reconstruction.
    :param true_rotations:      (Optional) True rotations of each projection-image to be used for comparison with
                                The estimated rotations.

    :return:                    Estimated reconstruction of the volume (aspire Volume) and estimated rotations (np array)
    """
    sym = supported_symmetry_group(sym) # Check provided symmetry type is valid
    if basis not in ("FFB", "Dirac"):
        raise ValueError(f"basis must be 'FFB' or 'Dirac', got '{basis}'")
    outvol_folder_name = os.path.dirname(outvol)  # Extract folder from outvol path
    if outvol_folder_name and not os.path.isdir(outvol_folder_name):
        raise FileNotFoundError( # There is a folder in the provided path but it is invalid
            f"Folder {outvol_folder_name} does not exist. Please create it first.\n"
        )

    ########################
    # Step 1: Cache file   #
    ########################

    cache_file_name = (
        ROTATIONS_CACHE_DIR / f"cache_{sym}_symmetry_resolution_{rotation_resolution}"
        f"_ntheta_{n_theta}_view_direc_{viewing_direction}_in_plane_{in_plane_rotation}.pkl"
    )
    local_cache_exists = cache_file_name.exists()
    if local_cache_exists:
        logger.info(f"Using existing local cache found: {cache_file_name}\n")
        check_cache_file(cache_file_name, sym, rotation_resolution, n_theta)
    else:
        logger.info(f"Creating cache file in folder: {cache_file_name.parent}\n")
        cache_config = (
            rotation_resolution,
            n_theta,
            viewing_direction,
            in_plane_rotation,
        )
        cache_file_name = cryo_create_rotations_cache(
            sym,
            cache_file_name,
            cache_config,
        )

    ###################################################################
    # Step 2: Load projections and downsample for rotation estimation #
    ###################################################################

    projs = load_projections_downsample(instack, ds_size=ds_rot_est, info=True)
    n_images, img_size = projs.shape[0], projs.shape[1]

    ##################################################
    # Step 3: Polar Fourier transform of projections #
    ##################################################

    logger.info("Computing the polar Fourier transform of projections")
    n_r = math.ceil(img_size * n_r_perc / 100)
    pft = PolarFT(img_size, nrad=n_r, ntheta=n_theta)
    pf = pft.transform(projs)  # pf has size (n_images) x (n_theta/2) x (n_rad),
    # with pf[:, :, 0] containing low frequency content and pf[:, :, -1] containing
    # high frequency content.
    pf[..., 0] = 0
    pf /= np.linalg.norm(pf, axis=2, ord=2)[..., np.newaxis]  # Normalize each ray.
    pf_full = PolarFT.half_to_full(
        pf
    )  # pf_full has size (n_images) x (n_theta) x (n_rad)

    logger.info(
        f"Polar Fourier transform of {pf_full.shape[0]} projections calculated. Each "
        f"contains {pf_full.shape[1]} rays with {pf_full.shape[2]} coefficients.\n"
    )

    ##############################################
    # Step 4: Computing the relative rotations   #
    ##############################################

    logger.info("Computing all relative rotations...\n")

    est_rel_rots = estimate_relative_rotations(
        pf_full, cache_file_name, sym, true_rotations=true_rotations,
    )
    logger.info("Done computing the relative rotations\n")
    logger.info(f"Upper block of relative rotations:\n\n{est_rel_rots[:14,:14]}\n\n")

    if true_rotations is not None:
        logger.info("Evaluating relative rotations against ground truth rotations...\n")
        rel_rots_dists = angular_distance_rel_rots(est_rel_rots, true_rotations, sym, cache_file_name)
        logger.info(f"Upper block of min angular distances: \n\n{rel_rots_dists[:14,:14]}\n\n")

    ##################################
    # Step 5: Rotation estimation    #
    ##################################

    logger.info("Estimating rotations")
    rots, cache_selected_data = estimate_rotations(sym, cache_file_name, est_rel_rots, n_images, n_theta, pf_full)

    ##########################################
    # Step 6: Estimated rotations evaluation #
    ##########################################

    if true_rotations is not None:
        logger.info("Comparing with known ground truth rotations")
        rots = mean_angular_distance_sym(rots, true_rotations, sym)

    logging.info(f"Cache selection data:\n {pd.DataFrame(cache_selected_data).T.to_string()}")

    if gui:
        if true_rotations is not None:
            create_projection_viewer_simulation(sym, cache_file_name, true_rotations, cache_selected_data)
        else:
            create_projection_viewer_class_avgs(sym, cache_file_name, projs.asnumpy(), cache_selected_data)

    ##################################
    #  Step 7: Volume reconstruction #
    ##################################

    logger.info("Reconstructing volume...\n")
    projs_reconstruction = load_projections_downsample(instack, ds_reconstruct, info=True)

    src = ArrayImageSource(projs_reconstruction, angles=Rotation(rots).angles, symmetry_group=sym)
    basis_obj = FFBBasis3D(src.L, dtype=src.dtype) if basis == "FFB" else DiracBasis3D(src.L, dtype=src.dtype)
    estimator = MeanEstimator(
        src=src,
        basis=basis_obj,
        maxiter=cg_max_iterations,
        checkpoint_iterations=None,
        preconditioner="none",
    )
    estimated_volume = estimator.estimate()

    ###########################
    # Step 8: Saving volume   #
    ###########################

    with mrcfile.new(outvol, overwrite=True) as mrc:
        mrc.set_data(estimated_volume)
        if voxel_size is not None:
            mrc.voxel_size = voxel_size
        mrc.update_header_from_data()
    logger.info(f"Volume saved to file: {outvol}")

    return estimated_volume, rots
