import numpy as np
import logging
import shutil

from aspire.noise import AnisotropicNoiseEstimator, CustomNoiseAdder
from aspire.operators import FunctionFilter
from aspire.source import Simulation
from aspire.volume import Volume

from cryosym.group_elements import group_elements
from cryosym.config import PROJECTIONS_DIR, RECONSTRUCTED_VOLUMES_DIR
from cryosym.utils import show
from cryosym.ab_initio_sym import cryo_abinitio_sym
from cryosym.volume_download.data_downloader import data_downloader

logger = logging.getLogger(__name__)

def ab_initio_simulation_demo(
    sym: str,
    num_imgs: int = 10,
    ds_rot_est: int = 70,
    ds_reconstruct: int = 120,
    noise_variance: float = 0.0,
    interactive: bool = True,
    estimate_noise: bool = False,
    resolution: str = "low",
    rotation_resolution=150,
    n_theta=360,
    cg_max_iterations=30,
    basis="Dirac",
    gui=True,
):

    volume_file = data_downloader(sym)

    if resolution == "high":
        in_plane_rotation = 2
        viewing_direction = 0.9985
    if resolution == "mid":
        in_plane_rotation = 3
        viewing_direction = 0.9975
    if resolution == "low":
        in_plane_rotation = 5
        viewing_direction = 0.996

    og_vol = Volume.load(volume_file)

    logger.info(
        "Original volume map data" f" shape: {og_vol.shape} dtype:{og_vol.dtype}"
    )
    voxel_size = (og_vol.resolution * og_vol.pixel_size) / ds_reconstruct

    # Plot projections on group elements
    # rots = group_elements(sym)
    # rots = np.float32(rots)
    # projections_est = og_vol.project(rots)
    # show(projections_est[:14], Title="Projections on group elements")

    def noise_function(x, y):
        alpha = 1
        beta = 1
        # White
        f1 = noise_variance
        # Violet-ish
        f2 = noise_variance * (x * x + y * y) / (og_vol.resolution * og_vol.resolution)
        return (alpha * f1 + beta * f2) / 2.0

    custom_noise = CustomNoiseAdder(noise_filter=FunctionFilter(noise_function))

    src = Simulation(
        n=num_imgs,
        vols=og_vol,
        offsets=0,
        amplitudes=1,
        noise_adder=custom_noise,
        # unique_filters=ctf_filters,
    )
    true_rotations = src.rotations

    file_name = f"simulated_{sym}_{num_imgs}_projections_{resolution}_ds_recon_{ds_reconstruct}_ds_rots_{ds_rot_est}_noise_{noise_variance}_basis_{basis}"
    projections_file_name = PROJECTIONS_DIR / f"{file_name}.mrc"

    output_dir = RECONSTRUCTED_VOLUMES_DIR / f"reconstructed_{file_name}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    reconstructed_vol_file_name = output_dir / f"reconstructed_{file_name}.mrc"

    if interactive:
        show(
            src.images[:10],
            Title=f"Simulated projection images (noise variance = {noise_variance})",
            save_path=output_dir / "simulated_projections.png",
        )

    # Estimate the noise and `Whiten` based on the estimated noise
    if estimate_noise:
        aniso_noise_estimator = AnisotropicNoiseEstimator(src)
        src = src.whiten(aniso_noise_estimator)
        if interactive:
            show(src.images[:10], Title="Whitened denoised images",
                 save_path=output_dir / "whitened_projections.png")

    src.images[:].save(projections_file_name, overwrite=True)

    estimated_volume, est_rots = cryo_abinitio_sym(
        sym,
        projections_file_name,
        reconstructed_vol_file_name,
        rotation_resolution=rotation_resolution,
        n_theta=n_theta,
        viewing_direction=viewing_direction,
        in_plane_rotation=in_plane_rotation,
        n_r_perc=50,
        ds_rot_est = ds_rot_est,
        ds_reconstruct = ds_reconstruct,
        cg_max_iterations=cg_max_iterations,
        basis=basis,
        true_rotations=true_rotations,
        voxel_size=voxel_size,
        gui=gui,
    )
    if interactive:
        est_rots = np.float32(est_rots)
        projections_est = estimated_volume.project(est_rots[:10])
        show(
            projections_est,
            Title="Projections from reconstructed volume and estimated rotations",
            save_path=output_dir / "reconstructed_projections.png",
        )
        # can be compared to those shown previously

    ds_v = og_vol.downsample(ds_reconstruct)

    # Save FSC plot into the output directory
    fsc_plot_path = output_dir / "fsc_plot.png"
    fsc, _ = estimated_volume.fsc(ds_v, cutoff=0.5, plot=str(fsc_plot_path))

    logger.info(
        f"Estimated FSC {fsc} Angstroms at {ds_v.pixel_size} Angstrom per pixel."
    )

    # Rename .mrc file and directory to include FSC score
    fsc_str = f"{fsc[0]:.2f}".replace(".", ",")
    reconstructed_vol_file_name.rename(reconstructed_vol_file_name.with_name(reconstructed_vol_file_name.stem + f"_FSC_{fsc_str}" + reconstructed_vol_file_name.suffix))
    output_dir_with_fsc = output_dir.with_name(output_dir.name + f"_FSC_{fsc_str}")
    if output_dir_with_fsc.exists():
        shutil.rmtree(output_dir_with_fsc)
    output_dir.rename(output_dir_with_fsc)

    logger.info(f"Saved reconstruction to directory: {output_dir_with_fsc}")


if __name__ == "__main__":

    for resolution in ['mid']:
        for num_imgs in [20]:
            for sym in ['O']:
                ab_initio_simulation_demo(
                            sym=sym,
                            num_imgs=num_imgs,
                            noise_variance=0,
                            ds_rot_est=79,
                            ds_reconstruct=121,
                            resolution=resolution,
                            interactive=True,
                            gui=False,
                        )


