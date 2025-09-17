import argparse
import os
from pathlib import Path

from .pipeline import run_structprep


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Structural MRI preprocessing for model pretraining"
    )

    p.add_argument("run", nargs="?", help="Run the pipeline", default="run")
    p.add_argument("--input-dir", required=True, help="Input BIDS-like directory")
    p.add_argument("--output-dir", required=True, help="Output derivatives directory")
    p.add_argument("--subjects", nargs="+", required=True, help="Subject IDs (sub-XXX)")
    p.add_argument("--sessions", nargs="*", default=None, help="Optional session IDs (ses-XXX)")
    p.add_argument(
        "--modalities",
        default="T1w,T2w,FLAIR",
        help="Comma-separated modalities to process",
    )

    # Processing flags
    p.add_argument("--no-n4", action="store_true", help="Disable N4 bias correction")
    p.add_argument("--no-norm", action="store_true", help="Disable intensity normalization")
    p.add_argument(
        "--isotropic",
        type=float,
        default=1.0,
        help="Isotropic voxel size for training export (mm)",
    )
    p.add_argument(
        "--shape",
        default="256x256",
        help="Canonical in-plane shape (HxW), e.g., 256x256. Depth (D) is preserved by default.",
    )
    # keep-depth defaults to True; provide explicit opt-out
    p.add_argument("--keep-depth", dest="keep_depth", action="store_true", default=True,
                   help="Preserve current slice count (D) when standardizing shape (default)")
    p.add_argument("--no-keep-depth", dest="keep_depth", action="store_false",
                   help="Allow crop/pad of depth (D) to match provided shape")
    p.add_argument(
        "--mask-aggressiveness",
        choices=["liberal", "medium", "conservative"],
        default="liberal",
        help="Brain mask dilation aggressiveness",
    )
    # Mask method fixed to FreeSurfer; keep hidden arg for compatibility
    p.add_argument(
        "--mask-method",
        default="freesurfer",
        choices=["freesurfer"],
        help="Masking uses FreeSurfer mri_watershed (-h 30 -atlas) + mri_binarize",
    )
    # Prefer explicit --fs-bin, else env FREESURFER_HOME_BIN, else $FREESURFER_HOME/bin
    default_fs_bin = os.environ.get("FREESURFER_HOME_BIN", "")
    if not default_fs_bin:
        fs_home = os.environ.get("FREESURFER_HOME", "")
        if fs_home:
            default_fs_bin = str(Path(fs_home) / "bin")

    p.add_argument(
        "--fs-bin",
        default=default_fs_bin,
        help="Path to FreeSurfer bin (optional; otherwise expect in PATH)",
    )

    # Concurrency
    p.add_argument("--n-jobs", type=int, default=4, help="Parallel workers")
    p.add_argument("--nprocs", type=int, default=os.cpu_count() or 1, help="Threads total")
    p.add_argument("--omp-nthreads", type=int, default=6, help="OMP threads per job")
    p.add_argument("--mem-mb", type=int, default=16000, help="Memory budget (MB)")

    p.add_argument("--dry-run", action="store_true", help="Do not write outputs")


    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.run != "run":
        parser.error("Only 'run' subcommand is supported")

    # Parse HxW and map to (D,H,W); D is a placeholder and ignored when keep_depth=True
    hw = tuple(int(x) for x in args.shape.lower().split("x"))
    if len(hw) != 2:
        parser.error("--shape must be HxW, e.g., 256x256")
    shape = (1, hw[0], hw[1])
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]

    run_structprep(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        subjects=args.subjects,
        sessions=args.sessions,
        modalities=modalities,
        n4=not args.no_n4,
        normalize=not args.no_norm,
        iso_mm=args.isotropic,
        out_shape=shape,
        keep_depth=args.keep_depth,
        mask_aggr=args.mask_aggressiveness,
        mask_method=args.mask_method,
        fs_bin=Path(args.fs_bin) if args.fs_bin else None,
        n_jobs=args.n_jobs,
        nprocs=args.nprocs,
        omp_nthreads=args.omp_nthreads,
        mem_mb=args.mem_mb,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
