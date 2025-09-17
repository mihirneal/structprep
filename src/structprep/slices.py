from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as ndi_zoom


@dataclass
class SliceSpec:
    count: int = 16
    start_thr: float = 0.08
    end_thr: float = 0.08
    target_size: Optional[Tuple[int, int]] = None  # (H, W) or None for no resize


def _brain_boundaries(mask: np.ndarray, thr_start: float, thr_end: float) -> Tuple[Optional[int], Optional[int], np.ndarray]:
    h, w, d = mask.shape
    coverage = np.zeros(d, dtype=np.float32)
    for z in range(d):
        slice_mask = mask[:, :, z] > 0
        coverage[z] = slice_mask.mean() if slice_mask.size > 0 else 0.0
    idx_start = np.where(coverage >= thr_start)[0]
    idx_end = np.where(coverage >= thr_end)[0]
    if idx_start.size == 0 or idx_end.size == 0:
        return None, None, coverage
    a = int(idx_start[0])
    b = int(idx_end[-1])
    if b < a:
        a, b = min(a, b), max(a, b)
    return a, b, coverage


def _indices_between(a: int, b: int, n: int) -> np.ndarray:
    return np.linspace(a, b, n, dtype=int)


def _resize_stack(stack: np.ndarray, target: Tuple[int, int], order: int) -> np.ndarray:
    t, h, w = stack.shape
    zh = target[0] / h
    zw = target[1] / w
    out = np.empty((t, target[0], target[1]), dtype=stack.dtype)
    for i in range(t):
        out[i] = ndi_zoom(stack[i], (zh, zw), order=order)
    return out


def extract_slices(
    vol_path: Path,
    mask_path: Path,
    spec: SliceSpec,
) -> Optional[tuple[np.ndarray, np.ndarray, list[int], dict]]:
    """Extract N axial slices from volume using mask coverage.

    Returns (volume_slices, mask_slices, indices, meta) or None if no brain found.
    """
    img = nib.load(str(vol_path))
    vol = img.get_fdata().astype(np.float32)
    msk = nib.load(str(mask_path)).get_fdata().astype(bool)
    if vol.shape[:3] != msk.shape[:3]:
        # Resampling should have matched shapes before calling
        raise ValueError(f"Volume/mask shape mismatch: {vol.shape} vs {msk.shape}")

    # Clamp thresholds to [0,1]
    z0, z1, cov = _brain_boundaries(
        msk,
        float(max(0.0, min(1.0, spec.start_thr))),
        float(max(0.0, min(1.0, spec.end_thr))),
    )
    if z0 is None or z1 is None:
        return None

    idx = _indices_between(z0, z1, spec.count)
    vol_s = np.stack([vol[:, :, z] for z in idx], axis=0)
    msk_s = np.stack([msk[:, :, z] for z in idx], axis=0)

    if spec.target_size and (vol_s.shape[1] != spec.target_size[0] or vol_s.shape[2] != spec.target_size[1]):
        vol_s = _resize_stack(vol_s, spec.target_size, order=1)
        msk_s = _resize_stack(msk_s.astype(np.float32), spec.target_size, order=0) > 0.5
        msk_s = msk_s.astype(np.uint8)

    meta = {
        "input_volume": str(vol_path),
        "input_mask": str(mask_path),
        "indices": [int(i) for i in idx],
        "coverage_per_slice": cov.tolist(),
        "slices_shape": list(vol_s.shape),
    }
    return vol_s.astype(np.float32), msk_s.astype(np.uint8), [int(i) for i in idx], meta


def pack_npz(volume: np.ndarray, mask: np.ndarray, meta: dict) -> bytes:
    buf = BytesIO()
    np.savez_compressed(buf, volume=volume, mask=mask, meta=meta)
    return buf.getvalue()
