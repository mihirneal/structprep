structprep — Structural MRI preprocessing for model pretraining

Overview
- Structural-only preprocessing tailored for pretraining. No MNI normalization (optional later). Produces per-session, voxel-aligned volumes, a liberal brain mask, and a canonical training tensor.

Pipeline (per session)
- RAS orientation: Convert inputs to RAS canonical orientation.
- Target selection: Choose the highest spatial resolution image (min voxel volume; tie-break by largest matrix) as the session target.
- Affine coregistration: Align all other structural scans (e.g., T2w, FLAIR) to the target using FreeSurfer mri_coreg (default params). Save LTA.
- Resampling: Apply affine using FreeSurfer mri_vol2vol to resample onto the target grid.
- N4 bias correction (default on): N4 (SimpleITK) on aligned volumes.
- Brain mask: FreeSurfer mri_watershed (-h 30 -atlas) to estimate brain volume, then mri_binarize to produce a binary mask, optionally dilated (liberal/medium/conservative). Do not skull-strip; provide binary mask only.
- Intensity normalization (default on): robust z-score within brain mask with percentile clipping.
- Isotropic + shape standardization: resample to 1.0 mm isotropic and crop/pad in-plane to a canonical size (default 256x256). Depth (D) is preserved by default; use `--no-keep-depth` to change D.
- Transforms & logs: Saves LTA (mov→target) and inverse LTA (target→mov). Writes JSON sidecars for N4 bias correction and normalization with basic stats and QC flags.
- Optional slice extraction to WebDataset: run the separate tool after preprocessing to extract N axial slices (default 16) between brain boundaries (from the binary mask), optionally resize (e.g., 224x224), and write a session-level `.tar` shard with NPZ entries (`slices.npz`) per modality under `<output-dir>/wds/`.

Outputs layout (per session)
- Final (`anat/final/`):
  - `*_space-iso1mm_desc-train.nii.gz` (primary training tensors)
  - `<sub>_<ses>_desc-brain_mask_space-iso1mm.nii.gz` (binary mask aligned to training grid)
- Work (`anat/work/`):
  - `*_desc-ras.nii.gz` (+ JSON)
  - `*_space-sesTarget.nii.gz`
  - `*_space-sesTarget_desc-affineToTarget_xfm.lta` and `*_space-sesTarget_desc-targetToMov_xfm.lta`
  - `*_desc-biascorr.nii.gz` (+ `*_desc-biascorr.json`)
  - `*_desc-brain_mask.nii.gz`
  - `*_desc-norm.nii.gz` (+ `*_desc-norm.json`)
  - `*_space-iso1mm_desc-train_desc-qc_mosaic.png` (QC)

CLI
- Install deps with uv: `uv sync`
- Run: `uv run structprep run --input-dir /path/to/bids --output-dir /path/to/derivatives/structprep --subjects sub-001 sub-002 --n-jobs 4 --nprocs 32 --omp-nthreads 6`
- Convenience script (auto concurrency): `sh structprep/scripts/run_structprep.sh -i /path/to/bids -o /path/to/derivatives/structprep`
  - Auto-discovers subjects; saturates CPU by default (n_jobs = min(#subjects, cores), omp = cores/n_jobs).
  - Example: `sh structprep/scripts/run_structprep.sh -i /data/bids -o /data/derivatives/structprep -a liberal`
  - Keep depth (default) and standardize in-plane to 256×256: `sh structprep/scripts/run_structprep.sh -i /data/bids -o /data/derivatives/structprep -p 256x256`
- Build WebDataset after preprocessing:
  - `uv run structprep-make-wds --derivatives-dir /data/derivatives/structprep --slices-per-volume 16 --slice-size 224x224`
  - Or use helper: `sh structprep/scripts/make_wds.sh -d /data/derivatives/structprep -S 16 -Z 224x224`
    - Flags: `-M` modalities, `--shard-size`, `--prefix`, `--start-index`, `-g` to group per subject, `-c` coverage threshold (default 0.08)

Concurrency
- Flags: `--n-jobs`, `--nprocs`, `--omp-nthreads`, `--mem-mb` control parallelism similar to fMRIPrep. The runner parallelizes across subject×session units.

Requirements
- Python 3.10+
- FreeSurfer 7.4.x in PATH (`mri_coreg`, `mri_vol2vol`, `mri_watershed`, `mri_binarize`, `lta_convert`)
- No FSL dependency; masking uses FreeSurfer exclusively
- WebDataset Python package for `.tar` shard writing when `--extract-slices` is used
- Packages: nibabel, numpy, scipy, scikit-image, SimpleITK, matplotlib, tqdm
