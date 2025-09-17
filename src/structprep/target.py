from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np


def choose_session_target(ras_files: List[Path]) -> Path:
    """Pick the highest spatial resolution image as target.

    Criterion: minimal voxel volume (product of voxel sizes), tie-break by largest matrix size.
    """
    best = None
    best_voxvol = None
    best_mat = None
    for p in ras_files:
        img = nib.load(str(p))
        zooms = np.array(img.header.get_zooms()[:3])
        voxvol = float(np.prod(zooms))
        matsz = int(np.prod(img.shape[:3]))
        if (
            best is None
            or voxvol < best_voxvol
            or (np.isclose(voxvol, best_voxvol) and matsz > best_mat)
        ):
            best = p
            best_voxvol = voxvol
            best_mat = matsz
    assert best is not None
    return best

