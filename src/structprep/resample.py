from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk


def resample_isotropic(in_file: Path, iso_mm: float, out_dir: Path, dry_run: bool = False) -> Path:
    out_path = out_dir / in_file.name.replace(".nii.gz", f"_space-iso{iso_mm:g}mm.nii.gz")
    if dry_run:
        return out_path

    img = sitk.ReadImage(str(in_file))
    spacing = np.array(img.GetSpacing(), dtype=float)
    # Compute new size to preserve FOV
    new_spacing = np.array([iso_mm, iso_mm, iso_mm], dtype=float)
    new_size = np.round(np.array(img.GetSize()) * (spacing / new_spacing)).astype(int)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(tuple(new_spacing.tolist()))
    resampler.SetSize([int(x) for x in new_size.tolist()])
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    out = resampler.Execute(img)
    sitk.WriteImage(out, str(out_path))
    return out_path

def resample_isotropic_mask(in_file: Path, iso_mm: float, out_dir: Path, dry_run: bool = False) -> Path:
    """Resample mask to isotropic spacing using nearest neighbor."""
    out_path = out_dir / in_file.name.replace(".nii.gz", f"_space-iso{iso_mm:g}mm_desc-mask.nii.gz")
    if dry_run:
        return out_path

    img = sitk.ReadImage(str(in_file))
    spacing = np.array(img.GetSpacing(), dtype=float)
    new_spacing = np.array([iso_mm, iso_mm, iso_mm], dtype=float)
    new_size = np.round(np.array(img.GetSize()) * (spacing / new_spacing)).astype(int)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(tuple(new_spacing.tolist()))
    resampler.SetSize([int(x) for x in new_size.tolist()])
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    out = resampler.Execute(img)
    sitk.WriteImage(out, str(out_path))
    return out_path


def crop_or_pad_center(
    in_file: Path,
    target_shape: Tuple[int, int, int],
    out_dir: Path,
    dry_run: bool = False,
    keep_depth: bool = False,
) -> Path:
    out_path = out_dir / in_file.name.replace(".nii.gz", "_desc-train.nii.gz")
    if dry_run:
        return out_path

    img = nib.load(str(in_file))
    data = img.get_fdata()
    tgt = np.array(target_shape, dtype=int)
    cur = np.array(data.shape[:3], dtype=int)

    # Preserve current depth when requested or when target D is a placeholder (<=1)
    if keep_depth or tgt[0] <= 1:
        # Preserve current depth; only standardize H and W
        tgt[0] = cur[0]

    # Pad
    pad = np.maximum(tgt - cur, 0)
    pad_before = (pad // 2).astype(int)
    pad_after = (pad - pad_before).astype(int)
    if keep_depth or tgt[0] == cur[0]:
        pad_before[0] = 0
        pad_after[0] = 0
    if pad.sum() > 0:
        data = np.pad(
            data,
            ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2])),
            mode="constant",
            constant_values=0,
        )

    # Crop centered
    cur2 = np.array(data.shape[:3], dtype=int)
    start = np.maximum((cur2 - tgt) // 2, 0)
    end = start + tgt
    if keep_depth or tgt[0] == cur2[0]:
        # Only crop H and W; keep full depth
        data = data[:, start[1] : end[1], start[2] : end[2]]
    else:
        data = data[start[0] : end[0], start[1] : end[1], start[2] : end[2]]

    out = nib.Nifti1Image(data.astype(np.float32), img.affine, img.header)
    nib.save(out, str(out_path))
    return out_path
