"""
Module responsible for downloading specified molecular volume data from the
Electron Microscopy Data Bank (EMD) based on known structural symmetries.
"""

import logging
import numpy as np

from aspire.volume import Volume
from aspire.utils import Rotation

from cryosym.config import cryo_fetch_emdID
from cryosym.config import DOWNLOADED_VOLUMES_DIR
from cryosym.volume_download.symmetry_group_conventions import (
    identify_symmetry_convention, SYMMETRY_CONVENTIONS,
    STANDARD_CONVENTIONS, conjugation_to_standard, sym_family,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", force=True)

EMD_ID_MAP = {
    "T"  : 10835,
    "O"  : 4905,  # 4905, 21173
    "I"  : 3952,  # 3952, 28027, 10295
    "C2" : 2824,
    "C3" : 2484, # 14621, 2484
    "C4" : 5778,
    "C5" : 3645,
    "C11": 6458,
    "D2" : 2984, # 2984, 8743, 20907
    "D3" : 14803, # 14803, 20016
    "D4" : 22308,
    "D5" : 28025,
    "D6" : 22358,
    "D7" : 6287,  # 6287, 25869
    "D8" : 9571,
    "D10": 10920,
    "D11": 21140,
    "D12": 11023,
    "D13": 20619,
}


def _rotate_volume_in_place(filename, M):
    """Load an MRC volume, rotate it by matrix M, and overwrite the file."""
    vol = Volume.load(str(filename))
    rot = Rotation(M.astype(np.float32).reshape(1, 3, 3))
    vol = vol.rotate(rot)
    vol.save(str(filename), overwrite=True)


def data_downloader(sym: str, emd_id: int | None = None, rotate_to_standard_convention=True) -> str:
    """
    Download molecule volume from EMD for different symmetry types.

    :param sym: symmetry type, e.g., 'T', 'O', 'I', 'D2'
    :param emd_id: explicit EMD ID to use; if None, looks up EMD_ID_MAP
    :param rotate_to_standard_convention:
    :return:    filename of downloaded file, mrc file
    """
    if emd_id is None:
        emd_id = EMD_ID_MAP.get(sym)
    filename = DOWNLOADED_VOLUMES_DIR / f"{sym}_sym_emd{emd_id}.mrc"
    if not filename.exists():
        logger.info("Downloading: %s", filename)
        cryo_fetch_emdID(emd_id, str(filename))
        family = sym_family(sym)
        if rotate_to_standard_convention and family in SYMMETRY_CONVENTIONS:
            logger.info("Checking symmetry orientation convention: %s", sym)
            convention = identify_symmetry_convention(str(filename), sym)
            standard = STANDARD_CONVENTIONS[family]
            logger.info("Identified convention for %s: %s", sym, convention)
            if convention != standard:
                logger.info("Rotating volume from %s to %s", convention, standard)
                M = conjugation_to_standard(convention, sym)
                _rotate_volume_in_place(filename, M)
                logger.info("Volume rotated and saved to %s", filename)
    return str(filename)


def download_all():
    """Downloads EMD volumes for a predefined set of symmetries."""
    for sym in EMD_ID_MAP.keys():
        data_downloader(sym)

if __name__ == "__main__":
    download_all()
