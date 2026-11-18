import json
import os
import shutil
import mrcfile
import logging

import numpy as np
from aspire.image import Image
from aspire.volume import Volume

from cryosym.ab_initio_sym import cryo_abinitio_sym
from cryosym.config import DOWNLOADED_VOLUMES_DIR, PROJECTIONS_DIR, RECONSTRUCTED_VOLUMES_DIR, CLASS_AVERAGES_DIR
from cryosym.fsc import plot_fsc
from cryosym.sym_align import align_sym
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
    "D2": 2984,
    "C5": 3645,
    "C4": 8511,
}
class_avg_map = {
    "T": "10389",
    "O": "10272",
    "I": "10205",
    "D7": "10025",
    "D4": "10502",
    "D3": "12036",
    "D2": "10061",
    "C5": "10089",
    "C4": "10081",
}

def class_averages_reconstruction(
    sym, n_projs, resolution="low", gui=True, ds_rot_est=70, ds_reconstruct=120, basis="Dirac", cyclic_step=5.0,
):
    """
    Ab-initio reconstruction of `sym` from its class averages and FSC against the reference map.

    `resolution` sets the rotations cache of Dn, T, O and I; for Cn, which uses no cache, the grid spacing is
    `cyclic_step` instead (in degrees, see cryo_abinitio_sym), and it is added to the name of the output directory
    when it is not the default 5.
    """
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

    is_cyclic = sym[0] == "C"
    step_tag = f"_step_{cyclic_step:g}" if is_cyclic and cyclic_step != 5.0 else ""
    output_dir_name = f"class_avgs_{sym}_{class_avg_map[sym]}_{n_projs}_{ds_reconstruct}_{resolution}{step_tag}_basis_{basis}"
    output_dir = RECONSTRUCTED_VOLUMES_DIR / output_dir_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    show(projs[:10], Title="Class Averages", save_path=output_dir / "class_averages.png")

    reconstruction_file_name = output_dir / f"{output_dir_name}.mrc"

    ref_vol_path = data_downloader(sym, emd_id=ref_emd_map[sym])
    og_vol = Volume.load(ref_vol_path)
    voxel_size = (og_vol.resolution * og_vol.pixel_size) / ds_reconstruct

    if resolution == "high":
        in_plane_rotation = 2
        viewing_direction = 0.9985
    if resolution == "mid":
        in_plane_rotation = 3
        viewing_direction = 0.9975
    if resolution == "low":
        in_plane_rotation = 5
        viewing_direction = 0.996

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
        cyclic_step=cyclic_step,
    )
    ds_v = og_vol.downsample(ds_reconstruct)
    # One file per run, so that runs in parallel do not overwrite or delete each other's reference
    outvol = DOWNLOADED_VOLUMES_DIR / f"og_vol_downsampled_{output_dir_name}.mrc"
    with mrcfile.new(outvol, overwrite=True) as mrc:
        mrc.set_data(ds_v)
        mrc.voxel_size = ds_v.pixel_size
        mrc.update_header_from_data()

    aligned_stem = f"class_avgs_aligned_{sym}_{class_avg_map[sym]}_{n_projs}_ds_recon_{ds_reconstruct}_ds_rots_{ds_rot_est}_{resolution}{step_tag}_basis_{basis}"
    emalign_file_name = output_dir / f"emalign_{aligned_stem}.mrc"
    emalign_mrc_save(outvol, reconstruction_file_name, emalign_file_name)
    os.remove(outvol)

    fsc = align_and_evaluate(sym, ds_v, reconstruction_file_name, emalign_file_name, output_dir / f"{aligned_stem}.mrc")
    output_dir_with_fsc = _rename_with_fsc(output_dir, output_dir_name, output_dir / f"{aligned_stem}.mrc", fsc)
    logger.info(f"Saved reconstruction to directory: {output_dir_with_fsc}")


def align_and_evaluate(sym, ref, rec_file, emalign_file, aligned_file):
    """
    Align the reconstruction in `rec_file` with `ref` (aspire Volume on the same grid) with cryosym.sym_align, with the
    EMalign result in `emalign_file` as an extra candidate. Saves the aligned map to `aligned_file`, the FSC plot and
    the chosen alignment (alignment.json) to its directory, and returns the resolution at FSC 0.5 (a list of one).
    """
    output_dir = aligned_file.parent
    rec = Volume.load(rec_file).asnumpy()[0]
    emalign = Volume.load(emalign_file).asnumpy()[0]
    aligned, info = align_sym(ref.asnumpy()[0], rec, sym, extra={"EMalign": emalign})
    aligned = aligned.astype(np.float32)
    with mrcfile.new(aligned_file, overwrite=True) as mrc:
        mrc.set_data(aligned)
        mrc.voxel_size = ref.pixel_size
    fsc = plot_fsc(ref, Volume(aligned), cutoff=0.5, path=output_dir / "fsc_plot.png")
    info["fsc_0.5"] = fsc[0]
    info["fsc_0.5_emalign"] = plot_fsc(ref, Volume(emalign), cutoff=0.5)[0]
    (output_dir / "alignment.json").write_text(json.dumps(info, indent=2))
    logger.info(f"Estimated FSC {fsc} Angstroms at {ref.pixel_size} Angstrom per pixel "
                f"(EMalign alone: {info['fsc_0.5_emalign']:.2f}).")
    return fsc


def _rename_with_fsc(output_dir, output_dir_name, aligned_file, fsc):
    """Add the resolution to the names of the aligned map and of the output directory; returns the new directory."""
    fsc_str = f"{fsc[0]:.2f}".replace(".", ",")
    aligned_file.rename(aligned_file.with_name(aligned_file.stem + f"_FSC_{fsc_str}" + aligned_file.suffix))
    output_dir_with_fsc = output_dir.with_name(output_dir_name + f"_FSC_{fsc_str}")
    if output_dir_with_fsc != output_dir:
        if output_dir_with_fsc.exists():
            shutil.rmtree(output_dir_with_fsc)
        output_dir.rename(output_dir_with_fsc)
    return output_dir_with_fsc


if __name__ == "__main__":

    for sym in ['I', 'O', 'T']:
        class_averages_reconstruction(
            sym, 50, resolution="low", ds_rot_est=75, ds_reconstruct=120, gui=False
        )

