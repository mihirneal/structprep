from pathlib import Path

import shutil
import subprocess
import numpy as np
import nibabel as nib
from skimage.morphology import ball, binary_dilation


def _dilate_mask(mask: np.ndarray, aggressiveness: str) -> np.ndarray:
    # Optional dilation to meet requested aggressiveness
    radius = {"liberal": 3, "medium": 2, "conservative": 1}.get(aggressiveness, 3)
    if radius <= 0:
        return mask
    return binary_dilation(mask, ball(radius))


def _ensure_cmd(cmd: str):
    if not shutil.which(cmd):
        raise RuntimeError(f"Required command not found on PATH: {cmd}")


def _run(cmd: list[str]):
    subprocess.run(cmd, check=True)


def _fs_brain_mask(in_file: Path, out_path: Path, tmp_dir: Path, dry_run: bool = False) -> None:
    """FreeSurfer mask via mri_watershed (-h 30 -atlas) then mri_binarize.

    Steps:
    - mri_watershed -h 30 -atlas <in> <tmp>.mgz
    - mri_binarize --i <tmp>.mgz --min 1 --o <out>.nii.gz
      (use --min to support FS versions without --binarize flag)
    """
    _ensure_cmd("mri_watershed")
    _ensure_cmd("mri_binarize")

    tmp_mgz = tmp_dir / (out_path.name.replace(".nii.gz", ".mgz"))

    if not dry_run:
        # Watershed with requested tuning
        _run(["mri_watershed", "-h", "30", "-atlas", str(in_file), str(tmp_mgz)])
        # Binarize mask using intensity threshold (>0) directly to NIfTI
        _run(["mri_binarize", "--i", str(tmp_mgz), "--min", "1", "--o", str(out_path)])


# Removed FSL BET implementation per updated design (FreeSurfer-only)


def make_brain_mask(
    in_file: Path,
    out_dir: Path,
    aggressiveness: str = "liberal",
    dry_run: bool = False,
    method: str = "freesurfer",
    fs_bin: Path | None = None,
) -> Path:
    """Create brain mask using FreeSurfer only (mri_watershed + mri_binarize).

    - method: kept for backward compatibility, must be 'freesurfer'
    - aggressiveness: optional post-dilation to widen mask margins
    """
    # Allow FS bin path to be prepended to PATH (optional convenience)
    if fs_bin and fs_bin.exists():
        os_path = str(fs_bin)
        if os_path not in (shutil.which("mri_watershed") or ""):
            import os

            os.environ["PATH"] = os_path + ":" + os.environ.get("PATH", "")

    # Resolve method
    chosen = method.lower()
    if chosen != "freesurfer":
        raise ValueError("Only 'freesurfer' mask method is supported")

    out_path = out_dir / in_file.name.replace(".nii.gz", "_desc-brain_mask.nii.gz")
    tmp_dir = out_dir

    if not dry_run:
        _fs_brain_mask(in_file, out_path, tmp_dir, dry_run=False)

        # Optional dilation step to adjust aggressiveness
        img = nib.load(str(out_path))
        data = (img.get_fdata() > 0).astype(bool)
        data = _dilate_mask(data, aggressiveness)
        nib.save(nib.Nifti1Image(data.astype(np.uint8), img.affine, img.header), str(out_path))

    return out_path
