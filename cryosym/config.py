"""
Configuration file for directory paths using pathlib.
All paths are defined relative to the project root to ensure code works regardless of CWD.
Also exports emalign utilities so callers don't need to know about the `src` pip package (EMalign).
"""
from pathlib import Path

# Get the project root directory (parent of cryosym/)
PROJECT_ROOT = Path(__file__).parent.parent

# Define directory paths relative to project root
# These directories will be created if they don't exist when accessed
DOWNLOADED_VOLUMES_DIR = PROJECT_ROOT / "data" / "downloaded_volumes"
RECONSTRUCTED_VOLUMES_DIR = PROJECT_ROOT / "data" / "reconstructed_volumes"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
ROTATIONS_CACHE_DIR = PROJECT_ROOT / "data" / "rotations_cache"
CLASS_AVERAGES_DIR = PROJECT_ROOT / "data" / "class_averages"

# Ensure directories exist (create them if they don't)
DOWNLOADED_VOLUMES_DIR.mkdir(parents=True, exist_ok=True)
RECONSTRUCTED_VOLUMES_DIR.mkdir(parents=True, exist_ok=True)
PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)
ROTATIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CLASS_AVERAGES_DIR.mkdir(parents=True, exist_ok=True)

# Export emalign utilities for external use
from src.read_write import read_mrc
from src.align_volumes_3d import align_volumes
from src.cryo_fetch_emdID import cryo_fetch_emdID


