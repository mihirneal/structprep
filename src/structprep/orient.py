from pathlib import Path
from typing import Tuple

import nibabel as nib


def to_ras(in_file: Path, out_dir: Path, write: bool = True) -> Tuple[Path, dict]:
    """Reorient to closest RAS canonical space.

    Returns output path and minimal metadata.
    """
    img = nib.load(str(in_file))
    ras = nib.as_closest_canonical(img)
    out_name = in_file.name.replace(".nii.gz", "_desc-ras.nii.gz")
    out_path = out_dir / out_name
    if write:
        nib.save(ras, str(out_path))
    meta = {
        "InputFile": str(in_file),
        "RAS": True,
        "OriginalShape": list(img.shape),
    }
    return out_path, meta

