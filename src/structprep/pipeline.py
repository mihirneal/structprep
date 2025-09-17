import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .bids import find_sessions, list_modality_files
from .orient import to_ras
from .target import choose_session_target
from .register import coregister_affine, resample_with_lta, ensure_fs_tools, invert_lta
from .bias import n4_bias_correct
from .mask import make_brain_mask
from .intensity import robust_normalize
from .resample import resample_isotropic, crop_or_pad_center, resample_isotropic_mask
from .qc import save_mosaic


@dataclass
class ProcConfig:
    input_dir: Path
    output_dir: Path
    modalities: list[str]
    n4: bool
    normalize: bool
    iso_mm: float
    out_shape: tuple[int, int, int]
    keep_depth: bool
    mask_aggr: str
    mask_method: str
    fs_bin: Optional[Path]
    omp_nthreads: int
    dry_run: bool
    


def run_structprep(
    input_dir: Path,
    output_dir: Path,
    subjects: Iterable[str],
    sessions: Optional[Iterable[str]],
    modalities: list[str],
    n4: bool,
    normalize: bool,
    iso_mm: float,
    out_shape: tuple[int, int, int],
    keep_depth: bool,
    mask_aggr: str,
    mask_method: str,
    fs_bin: Optional[Path],
    n_jobs: int,
    nprocs: int,
    omp_nthreads: int,
    mem_mb: int,
    dry_run: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_fs_tools(fs_bin)

    # Environment tuning to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", str(omp_nthreads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    cfg = ProcConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        modalities=modalities,
        n4=n4,
        normalize=normalize,
        iso_mm=iso_mm,
        out_shape=out_shape,
        keep_depth=keep_depth,
        mask_aggr=mask_aggr,
        mask_method=mask_method,
        fs_bin=fs_bin,
        omp_nthreads=omp_nthreads,
        dry_run=dry_run,
    )

    work = []
    for sub in subjects:
        ses_list = find_sessions(input_dir, sub) if not sessions else list(sessions)
        for ses in ses_list:
            work.append((sub, ses))

    if not work:
        print("No subject/session work items found.")
        return

    max_workers = max(1, n_jobs)
    # Avoid multiprocessing overhead/permissions when single-threaded
    if max_workers == 1:
        for sub, ses in work:
            try:
                process_session(cfg, sub, ses)
                print(f"✓ Completed {sub} {ses}")
            except Exception as e:
                print(f"✗ Failed {sub} {ses}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(process_session, cfg, sub, ses): (sub, ses) for sub, ses in work}
            for fut in as_completed(futs):
                sub, ses = futs[fut]
                try:
                    fut.result()
                    print(f"✓ Completed {sub} {ses}")
                except Exception as e:
                    print(f"✗ Failed {sub} {ses}: {e}")


def process_session(cfg: ProcConfig, sub: str, ses: str):
    in_anat = Path(cfg.input_dir) / sub / ses / "anat"
    if not in_anat.is_dir():
        raise FileNotFoundError(f"Missing anat directory: {in_anat}")

    # Discover candidate files per modality
    files = list_modality_files(in_anat, cfg.modalities)
    if not files:
        raise FileNotFoundError(f"No modalities found in {in_anat}")

    out_anat = Path(cfg.output_dir) / sub / ses / "anat"
    out_anat_final = out_anat / "final"
    out_anat_work = out_anat / "work"
    out_anat_final.mkdir(parents=True, exist_ok=True)
    out_anat_work.mkdir(parents=True, exist_ok=True)

    # RAS for all inputs
    ras_files = {}
    for mod, paths in files.items():
        ras_files[mod] = []
        for p in paths:
            ras_p, meta = to_ras(p, out_anat_work, write=not cfg.dry_run)
            ras_files[mod].append(ras_p)
            if not cfg.dry_run:
                with open(str(ras_p).replace(".nii.gz", "_desc-ras.json"), "w") as f:
                    json.dump(meta, f, indent=2)

    # Choose session target by highest spatial resolution
    all_ras = [rp for lst in ras_files.values() for rp in lst]
    target = choose_session_target(all_ras)

    # Coreg + resample onto target grid
    aligned = {}
    ltas = {}
    for mod, paths in ras_files.items():
        aligned[mod] = []
        for mov in paths:
            if Path(mov).resolve() == Path(target).resolve():
                # Target itself — copy as aligned output
                out_mov = str(mov).replace("_desc-ras.nii.gz", "_space-sesTarget.nii.gz")
                if not cfg.dry_run:
                    shutil.copy2(mov, out_mov)
                aligned[mod].append(Path(out_mov))
                # Create identity LTA sidecar for completeness
                lta_path = out_mov.replace("_space-sesTarget.nii.gz", "_space-sesTarget_desc-affineToTarget_xfm.lta")
                if not cfg.dry_run:
                    with open(lta_path, "w") as f:
                        f.write("# identity placeholder for target\n")
                ltas[(mov, target)] = Path(lta_path)
                continue

            lta = coregister_affine(mov=mov, ref=target, out_dir=out_anat_work, omp=cfg.omp_nthreads, dry_run=cfg.dry_run)
            out_mov = resample_with_lta(mov=mov, ref=target, lta=lta, out_dir=out_anat_work, interp="trilinear", dry_run=cfg.dry_run)
            aligned[mod].append(out_mov)
            ltas[(mov, target)] = lta
            # Also produce inverse LTA (target->mov) for convenience
            inv_path = Path(str(lta).replace("_desc-affineToTarget_xfm.lta", "_desc-targetToMov_xfm.lta"))
            invert_lta(lta, inv_path, dry_run=cfg.dry_run)

    # Determine reference for mask (prefer T1w, else target)
    ref_img = None
    if "T1w" in aligned and len(aligned["T1w"]) > 0:
        ref_img = aligned["T1w"][0]
    else:
        ref_img = Path(target).with_name(Path(target).name.replace("_desc-ras.nii.gz", "_space-sesTarget.nii.gz"))
        if not ref_img.exists():
            ref_img = aligned[next(iter(aligned))][0]

    # Bias correction and mask on ref
    mask_path = None
    if cfg.n4:
        ref_bias = n4_bias_correct(ref_img, out_dir=out_anat_work, dry_run=cfg.dry_run)
    else:
        ref_bias = ref_img

    mask_path = make_brain_mask(
        ref_bias,
        out_dir=out_anat_work,
        aggressiveness=cfg.mask_aggr,
        dry_run=cfg.dry_run,
        method=cfg.mask_method,
        fs_bin=cfg.fs_bin,
    )

    # Write a single binary mask into final/ matching the training grid
    mask_iso = None
    mask_final_path = None
    if not cfg.dry_run:
        mask_iso = resample_isotropic_mask(mask_path, iso_mm=cfg.iso_mm, out_dir=out_anat_work, dry_run=cfg.dry_run)
        tmp_mask = crop_or_pad_center(
            mask_iso,
            target_shape=cfg.out_shape,
            out_dir=out_anat_final,
            dry_run=cfg.dry_run,
            keep_depth=cfg.keep_depth,
        )
        # Rename to a clean, session-level name
        mask_final_path = out_anat_final / f"{sub}_{ses}_desc-brain_mask_space-iso{cfg.iso_mm:g}mm.nii.gz"
        try:
            shutil.move(str(tmp_mask), str(mask_final_path))
        except Exception:
            mask_final_path = Path(tmp_mask)

    # For each aligned image: optional N4, normalization, training export
    for mod, paths in aligned.items():
        for p in paths:
            cur = p
            if cfg.n4 and p != ref_img:
                cur = n4_bias_correct(cur, out_dir=out_anat_work, dry_run=cfg.dry_run)

            norm_p = cur
            if cfg.normalize:
                norm_p = robust_normalize(cur, mask_path, out_dir=out_anat_work, dry_run=cfg.dry_run)

            # Export training tensor: isotropic + crop/pad
            iso = resample_isotropic(norm_p, iso_mm=cfg.iso_mm, out_dir=out_anat_work, dry_run=cfg.dry_run)
            train = crop_or_pad_center(iso, target_shape=cfg.out_shape, out_dir=out_anat_final, dry_run=cfg.dry_run, keep_depth=cfg.keep_depth)

            # QC mosaic
            if not cfg.dry_run:
                save_mosaic(train, out_dir=out_anat_work)

    # Remove work mask to reduce clutter now that final mask exists
    if not cfg.dry_run:
        try:
            Path(mask_path).unlink(missing_ok=True)
        except Exception:
            pass
        if mask_iso is not None:
            try:
                Path(mask_iso).unlink(missing_ok=True)
            except Exception:
                pass
