import os
import shutil
import subprocess
from pathlib import Path


def ensure_fs_tools(fs_bin: Path | None):
    """Ensure FreeSurfer tools are available in PATH; prepend fs_bin if provided."""
    if fs_bin and fs_bin.exists():
        p = str(fs_bin)
        if p not in (shutil.which("mri_coreg") or ""):
            # Prepend to PATH for this process and children
            import os

            os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
    # Quick availability check
    if (not shutil.which("mri_coreg") or
        not shutil.which("mri_vol2vol") or
        not shutil.which("lta_convert")):
        raise RuntimeError(
            "FreeSurfer tools mri_coreg/mri_vol2vol/lta_convert not found in PATH. Set --fs-bin."
        )


def coregister_affine(mov: Path, ref: Path, out_dir: Path, omp: int, dry_run: bool) -> Path:
    """Estimate affine LTA using mri_coreg (default params)."""
    out_lta = out_dir / (
        mov.name.replace("_desc-ras.nii.gz", "_space-sesTarget_desc-affineToTarget_xfm.lta")
    )
    cmd = [
        "mri_coreg",
        "--mov",
        str(mov),
        "--ref",
        str(ref),
        "--reg",
        str(out_lta),
    ]
    env = {**os.environ, "OMP_NUM_THREADS": str(max(1, omp))}
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)
    return out_lta


def resample_with_lta(
    mov: Path,
    ref: Path,
    lta: Path,
    out_dir: Path,
    interp: str = "trilinear",
    dry_run: bool = False,
) -> Path:
    out_img = out_dir / mov.name.replace("_desc-ras.nii.gz", "_space-sesTarget.nii.gz")
    cmd = [
        "mri_vol2vol",
        "--mov",
        str(mov),
        "--targ",
        str(ref),
        "--lta",
        str(lta),
        "--o",
        str(out_img),
        "--interp",
        interp,
    ]
    if not dry_run:
        subprocess.run(cmd, check=True)
    return out_img


def invert_lta(in_lta: Path, out_lta: Path, dry_run: bool = False) -> Path:
    """Create inverse LTA using lta_convert -invert."""
    cmd = [
        "lta_convert",
        "-invert",
        "-inlta",
        str(in_lta),
        "-outlta",
        str(out_lta),
    ]
    if not dry_run:
        subprocess.run(cmd, check=True)
    return out_lta
