from pathlib import Path
from typing import Dict, List


def find_sessions(bids_root: Path, subject: str) -> List[str]:
    """Return session directories (ses-*) under a subject."""
    sub_dir = bids_root / subject
    if not sub_dir.exists():
        return []
    sessions = [p.name for p in sub_dir.iterdir() if p.is_dir() and p.name.startswith("ses-")]
    sessions.sort()
    return sessions


def list_modality_files(anat_dir: Path, modalities: List[str]) -> Dict[str, List[Path]]:
    """List NIfTI files for requested modalities in anat dir.

    Matches filenames like sub-*_ses-*_T1w.nii.gz etc.
    """
    out: Dict[str, List[Path]] = {m: [] for m in modalities}
    for m in modalities:
        out[m] = sorted(anat_dir.glob(f"*_" + m + ".nii.gz"))
    # Drop empty modalities
    out = {k: v for k, v in out.items() if v}
    return out

