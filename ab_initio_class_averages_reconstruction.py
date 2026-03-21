import os
import shutil
import mrcfile
import logging

from aspire.image import Image
from aspire.volume import Volume

from cryosym.ab_initio_sym import cryo_abinitio_sym
from cryosym.config import DOWNLOADED_VOLUMES_DIR, PROJECTIONS_DIR, RECONSTRUCTED_VOLUMES_DIR, CLASS_AVERAGES_DIR
from cryosym.utils import show, emalign_mrc_save
from cryosym.volume_download.data_downloader import data_downloader

logger = logging.getLogger(__name__)


ref_emd_map = {
    "T": 10835,
    "O": 4905,
    "I": 3952,
    "D7": 6287,
    "D4": 22308,
    "D3": 14803,
}
class_avg_map = {
    "T": "10389",
    "O": "10272",
    "I": "10205",
    "D7": "10025",
    "D4": "10502",
    "D3": "12036",
}

def class_averages_reconstruction(
    sym, n_projs, resolution="low", gui=True, ds_rot_est=70, ds_reconstruct=120, basis="Dirac",
):
    projections_file_name = CLASS_AVERAGES_DIR / f"{class_avg_map[sym]}.mrc"
    projs = Image.load(
        projections_file_name,
    )
    n_projs = min(projs.shape[0], n_projs)
    projs = projs[:n_projs]
    projections_file_name = (
        PROJECTIONS_DIR / f"class_averages_{class_avg_map[sym]}_n_{n_projs}.mrc"
    )
    projs.save(projections_file_name)

    output_dir_name = f"class_avgs_{sym}_{class_avg_map[sym]}_{n_projs}_{ds_reconstruct}_{resolution}_basis_{basis}"
    output_dir = RECONSTRUCTED_VOLUMES_DIR / output_dir_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    show(projs[:10], Title="Class Averages", save_path=output_dir / "class_averages.png")

    reconstruction_file_name = output_dir / f"{output_dir_name}.mrc"

    ref_vol_path = data_downloader(sym, emd_id=ref_emd_map[sym])
    og_vol = Volume.load(ref_vol_path)
    voxel_size = (og_vol.resolution * og_vol.pixel_size) / ds_reconstruct

    if resolution == "vhigh":
        in_plane_rotation = 1.8
        viewing_direction = 0.9988
    if resolution == "high":
        in_plane_rotation = 2
        viewing_direction = 0.9985
    if resolution == "mid":
        in_plane_rotation = 3
        viewing_direction = 0.9975
    if resolution == "low":
        in_plane_rotation = 5
        viewing_direction = 0.996
    if resolution == "vlow":
        in_plane_rotation = 8
        viewing_direction = 0.99

    estimated_volume, est_rots = cryo_abinitio_sym(
        sym,
        projections_file_name,
        reconstruction_file_name,
        ds_rot_est=ds_rot_est,
        ds_reconstruct=ds_reconstruct,
        rotation_resolution=150,
        n_theta=360,
        n_r_perc=50,
        viewing_direction=viewing_direction,
        in_plane_rotation=in_plane_rotation,
        cg_max_iterations=30,
        basis=basis,
        voxel_size=voxel_size,
        gui=gui,
    )
    ds_v = og_vol.downsample(ds_reconstruct)
    outvol = DOWNLOADED_VOLUMES_DIR / "og_vol_downsampled.mrc"
    with mrcfile.new(outvol, overwrite=True) as mrc:
        mrc.set_data(ds_v)
        mrc.voxel_size = ds_v.pixel_size
        mrc.update_header_from_data()

    aligned_file_name = output_dir / f"class_avgs_aligned_{sym}_{class_avg_map[sym]}_{n_projs}_ds_recon_{ds_reconstruct}_ds_rots_{ds_rot_est}_{resolution}_basis_{basis}.mrc"
    emalign_mrc_save(outvol, reconstruction_file_name, aligned_file_name)
    os.remove(outvol)

    sim_vol_aligned = Volume.load(aligned_file_name)

    fsc_plot_path = output_dir / "fsc_plot.png"
    fsc, _ = sim_vol_aligned.fsc(ds_v, cutoff=0.5, plot=str(fsc_plot_path))
    logger.info(
        f"Estimated FSC {fsc} Angstroms at {ds_v.pixel_size} Angstrom per pixel."
    )

    # Rename aligned .mrc file and directory to include FSC score
    fsc_str = f"{fsc[0]:.2f}".replace(".", ",")
    aligned_file_name.rename(aligned_file_name.with_name(aligned_file_name.stem + f"_FSC_{fsc_str}" + aligned_file_name.suffix))
    output_dir_with_fsc = output_dir.with_name(output_dir.name + f"_FSC_{fsc_str}")
    if output_dir_with_fsc.exists():
        shutil.rmtree(output_dir_with_fsc)
    output_dir.rename(output_dir_with_fsc)

    logger.info(f"Saved reconstruction to directory: {output_dir_with_fsc}")


if __name__ == "__main__":

    for sym in ['I', 'O', 'T']:
        class_averages_reconstruction(
            sym, 50, resolution="low", ds_rot_est=75, ds_reconstruct=120, gui=False
        )

