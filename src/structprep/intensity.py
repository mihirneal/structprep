from pathlib import Path
import json

import nibabel as nib
import numpy as np


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def robust_normalize(in_file: Path, mask_file: Path, out_dir: Path, dry_run: bool = False) -> Path:
    out_path = out_dir / in_file.name.replace(".nii.gz", "_desc-norm.nii.gz")
    if dry_run:
        return out_path

    img = nib.load(str(in_file))
    data = img.get_fdata().astype(np.float32)
    m = nib.load(str(mask_file)).get_fdata().astype(bool)
    if m.sum() < 10:
        # Fallback to whole-volume
        m = np.ones_like(data, dtype=bool)

    vals = data[m]
    lo, hi = np.percentile(vals, [0.5, 99.5])
    vals_clipped = np.clip(vals, lo, hi)
    mean = float(vals_clipped.mean())
    std = float(vals_clipped.std() + 1e-6)

    data = np.clip(data, lo, hi)
    data = (data - mean) / std

    out = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(out, str(out_path))

    # Stats + QC sidecar
    try:
        from nibabel.orientations import aff2axcodes

        ax = aff2axcodes(img.affine)
        lr_ok = ax[0] == "R"
    except Exception:
        lr_ok = True

    stats = {
        "input_path": str(in_file),
        "output_path": str(out_path),
        "mask_path": str(mask_file),
        "mask_voxels": int(m.sum()),
        "p0p5_p99p5": [float(lo), float(hi)],
        "mean_std": [mean, std],
        "median_MAD": [float(np.median(vals)), _mad(vals)],
        "lr_flip_suspect": not lr_ok,
        "warnings": [],
    }
    # Heuristic warnings
    vol_vox = int(np.prod(data.shape[:3]))
    frac = stats["mask_voxels"] / max(1, vol_vox)
    if stats["mask_voxels"] < 10000:
        stats["warnings"].append("Very small mask voxel count")
    if frac > 0.9:
        stats["warnings"].append("Mask covers >90% of volume")
    if std < 1e-4:
        stats["warnings"].append("Low intensity variance after normalization")

    with open(str(out_path).replace(".nii.gz", ".json"), "w") as f:
        json.dump(stats, f, indent=2)

    return out_path
