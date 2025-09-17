from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict

import webdataset as wds

from .slices import SliceSpec, extract_slices, pack_npz


def find_sessions(deriv_root: Path, subjects: Optional[Iterable[str]] = None, sessions: Optional[Iterable[str]] = None):
    subs = subjects or [p.name for p in deriv_root.glob("sub-*") if p.is_dir()]
    for sub in subs:
        ses_list = sessions or [p.name for p in (deriv_root / sub).glob("ses-*") if p.is_dir()]
        for ses in ses_list:
            yield sub, ses


def infer_modality(path: Path) -> str:
    name = path.name
    for m in ("T1w", "T2w", "FLAIR"):
        if m in name:
            return m
    return "UNK"


def parse_size(s: str) -> Tuple[int, int]:
    parts = [int(x) for x in s.lower().split("x")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must be HxW, e.g., 224x224")
    return parts[0], parts[1]


def make_wds(
    derivatives_dir: Path,
    out_dir: Path,
    slices_per_volume: int,
    slice_size: Optional[Tuple[int, int]],
    modalities: Optional[Iterable[str]] = None,
    subjects: Optional[Iterable[str]] = None,
    sessions: Optional[Iterable[str]] = None,
    shard_size: int = 100,
    shard_prefix: str = "ADNI",
    start_index: int = 1,
    group_by_subject: bool = False,
    cov_start: float = 0.08,
    cov_end: float = 0.08,
    dry_run: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, str]] = []
    by_subject: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for sub, ses in find_sessions(derivatives_dir, subjects, sessions):
        anat = derivatives_dir / sub / ses / "anat"
        if not anat.is_dir():
            continue

        # Final mask and training tensors only
        masks = sorted((anat / "final").glob("*_desc-brain_mask_space-iso*mm.nii.gz"))
        if not masks:
            print(f"No brain mask in {anat}, skipping {sub} {ses}")
            continue
        mask_path = masks[0]

        trains = [p for p in sorted((anat / "final").glob("*_desc-train.nii.gz")) if "_desc-mask_" not in p.name]
        if not trains:
            print(f"No training tensors in {anat}, skipping {sub} {ses}")
            continue

        # Optional modality filter
        if modalities:
            mods_set = {m.strip() for m in modalities if m.strip()}
            trains = [p for p in trains if infer_modality(p) in mods_set]
            if not trains:
                continue

        for t in trains:
            mod = infer_modality(t)
            item = {"sub": sub, "ses": ses, "mod": mod, "train": str(t), "mask": str(mask_path)}
            samples.append(item)
            by_subject[sub].append(item)

    if dry_run:
        total = len(samples)
        shards = (total + max(1, shard_size) - 1) // max(1, shard_size)
        print(f"[dry-run] would write {total} samples across {shards} shards with prefix {shard_prefix}_NNN.tar")
        return

    def write_shard(shard_idx: int, items: List[Dict[str, str]]):
        if not items:
            return
        shard_path = out_dir / f"{shard_prefix}_{shard_idx:03d}.tar"
        with wds.TarWriter(str(shard_path)) as sink:
            for it in items:
                spec = SliceSpec(count=slices_per_volume, target_size=slice_size, start_thr=cov_start, end_thr=cov_end)
                vol = Path(it["train"])  # final train volume
                msk = Path(it["mask"])   # final mask
                out = extract_slices(vol, msk, spec)
                if out is None:
                    print(f"No brain coverage for {vol}, skipping")
                    continue
                vol_s, msk_s, idx, meta = out
                meta.update({
                    "subject": it["sub"],
                    "session": it["ses"],
                    "modality": it["mod"],
                    "train_path": it["train"],
                    "mask_path": it["mask"],
                })
                key = Path(it["train"]).name.replace(".nii.gz", "")
                sink.write({"__key__": key, "slices.npz": pack_npz(vol_s, msk_s, meta)})

    idx = start_index
    if group_by_subject:
        cur: List[Dict[str, str]] = []
        for sub in sorted(by_subject.keys()):
            subj_items = by_subject[sub]
            if cur and (len(cur) + len(subj_items) > shard_size):
                write_shard(idx, cur)
                idx += 1
                cur = []
            if len(subj_items) > shard_size and not cur:
                write_shard(idx, subj_items)
                idx += 1
            else:
                cur.extend(subj_items)
        if cur:
            write_shard(idx, cur)
            idx += 1
    else:
        for i in range(0, len(samples), shard_size):
            write_shard(idx, samples[i : i + shard_size])
            idx += 1


def main():
    ap = argparse.ArgumentParser(description="Build WebDataset shards of axial slices from preprocessed structprep outputs (final only)")
    ap.add_argument("--derivatives-dir", required=True, help="structprep output directory (contains sub-*/ses-*/anat)")
    ap.add_argument("--out-dir", default="", help="Output directory for WebDataset shards (default: <derivatives-dir>/wds)")
    ap.add_argument("--slices-per-volume", type=int, default=16, help="Number of slices per volume")
    ap.add_argument("--slice-size", type=parse_size, default=None, help="Optional HxW resize for slices (e.g., 224x224)")
    ap.add_argument("--modalities", default="T1w,FLAIR", help="CSV of modalities to include (e.g., T1w or T1w,FLAIR)")
    ap.add_argument("--subjects", nargs="*", default=None, help="Optional subject filter (list)")
    ap.add_argument("--sessions", nargs="*", default=None, help="Optional session filter (list)")
    ap.add_argument("--shard-size", type=int, default=100, help="Target samples per shard (default: 100)")
    ap.add_argument("--prefix", default="ADNI", help="Shard filename prefix (default: ADNI)")
    ap.add_argument("--start-index", type=int, default=1, help="Starting shard index (default: 1)")
    ap.add_argument("--group-by-subject", action="store_true", help="Keep each subject's samples within the same shard when possible")
    ap.add_argument("--coverage-thresholds", default="0.08,0.08", help="Start,End mask coverage thresholds (0..1). E.g., 0.10,0.15")
    ap.add_argument("--dry-run", action="store_true", help="Do not write shards; print actions")
    args = ap.parse_args()

    deriv = Path(args.derivatives_dir)
    out = Path(args.out_dir) if args.out_dir else (deriv / "wds")

    # Parse thresholds string
    thr_s = args.coverage_thresholds
    parts = [p.strip() for p in thr_s.split(",") if p.strip()]
    if len(parts) == 1:
        cov_start = cov_end = float(parts[0])
    else:
        cov_start = float(parts[0]); cov_end = float(parts[1])
    cov_start = max(0.0, min(1.0, cov_start))
    cov_end = max(0.0, min(1.0, cov_end))

    make_wds(
        derivatives_dir=deriv,
        out_dir=out,
        slices_per_volume=args.slices_per_volume,
        slice_size=args.slice_size,
        modalities=[m.strip() for m in args.modalities.split(",") if m.strip()],
        subjects=args.subjects,
        sessions=args.sessions,
        shard_size=max(1, args.shard_size),
        shard_prefix=args.prefix,
        start_index=max(0, args.start_index),
        group_by_subject=args.group_by_subject,
        cov_start=cov_start,
        cov_end=cov_end,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
