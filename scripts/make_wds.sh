#!/usr/bin/env sh
set -eu

usage() {
  cat <<'USAGE'
Build WebDataset shards of axial slices from structprep derivatives.

Usage:
  structprep/scripts/make_wds.sh [options]

Options:
  -d, --derivatives-dir  Path to structprep output (default: /home/mihirneal/ADNI_struct)
  -o, --out-dir          Output directory for shards (default: <derivatives-dir>/wds)
  -S, --slices-per-vol   Number of slices per volume (default: 16)
  -Z, --slice-size       Optional resize HxW for slices (e.g., 224x224)
  -M, --modalities       Modalities to include (CSV). Default: T1w,FLAIR
  --shard-size           Target samples per shard (default: 100)
  --prefix               Shard filename prefix (default: ADNI)
  --start-index          Starting shard index (default: 1)
  -g, --group-by-subject Keep each subject within a shard when possible
  -c, --coverage-thr     Mask coverage thresholds START,END (default: 0.08,0.08)
  -s, --subjects         Quoted space-separated subject IDs (e.g., "sub-001 sub-002")
  -e, --sessions         Quoted space-separated session IDs (e.g., "ses-01 ses-02")
      --dry-run          Do not write shards; print actions
  -h, --help             Show this help

Notes:
  - Run this after structprep preprocessing completes.
  - Writes one shard per session: <out-dir>/sub-XXX_ses-YYY.tar
USAGE
}

# Defaults
DERIV_DIR="/home/mihirneal/ADNI_struct"
OUT_DIR="/home/mihirneal/ADNI_wds"
SLICES_PER_VOL="16"
SLICE_SIZE="256x256"
MODALITIES="T1w"
SUBJECTS=""
SESSIONS=""
DRY_RUN=0
SHARD_SIZE=40
PREFIX="ADNI"
START_INDEX=1
GROUP_BY_SUBJ=0
COVERAGE_THR="0.10,0.20"

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      -d|--derivatives-dir) DERIV_DIR="$2"; shift 2;;
      -o|--out-dir) OUT_DIR="$2"; shift 2;;
      -S|--slices-per-vol) SLICES_PER_VOL="$2"; shift 2;;
      -Z|--slice-size) SLICE_SIZE="$2"; shift 2;;
      -M|--modalities) MODALITIES="$2"; shift 2;;
      --shard-size) SHARD_SIZE="$2"; shift 2;;
      --prefix) PREFIX="$2"; shift 2;;
      --start-index) START_INDEX="$2"; shift 2;;
      -g|--group-by-subject) GROUP_BY_SUBJ=1; shift 1;;
      -c|--coverage-thr) COVERAGE_THR="$2"; shift 2;;
      -s|--subjects)
        shift
        while [ $# -gt 0 ]; do
          case "$1" in -*) break;; esac
          if [ -n "$SUBJECTS" ]; then SUBJECTS="$SUBJECTS $1"; else SUBJECTS="$1"; fi
          shift
        done
        ;;
      -e|--sessions)
        shift
        while [ $# -gt 0 ]; do
          case "$1" in -*) break;; esac
          if [ -n "$SESSIONS" ]; then SESSIONS="$SESSIONS $1"; else SESSIONS="$1"; fi
          shift
        done
        ;;
      --dry-run) DRY_RUN=1; shift 1;;
      -h|--help) usage; exit 0;;
      --) shift; break;;
      *) echo "Unknown argument: $1" >&2; usage; exit 1;;
    esac
  done
}

parse_args "$@"

if [ -z "$DERIV_DIR" ]; then
  echo "ERROR: --derivatives-dir is required" >&2
  usage
  exit 1
fi

if [ ! -d "$DERIV_DIR" ]; then
  echo "ERROR: derivatives dir not found: $DERIV_DIR" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "WARNING: 'uv' not found on PATH; attempting to run via python -m. Consider installing uv for reproducibility." >&2
fi

# Build argument list for structprep-make-wds
set -- \
  --derivatives-dir "$DERIV_DIR" \
  --slices-per-volume "$SLICES_PER_VOL" \
  --modalities "$MODALITIES" \
  --shard-size "$SHARD_SIZE" \
  --prefix "$PREFIX" \
  --start-index "$START_INDEX" \
  --coverage-thresholds "$COVERAGE_THR"

if [ -n "$OUT_DIR" ]; then
  set -- "$@" --out-dir "$OUT_DIR"
fi
if [ -n "$SLICE_SIZE" ]; then
  set -- "$@" --slice-size "$SLICE_SIZE"
fi
if [ -n "$SUBJECTS" ]; then
  set -- "$@" --subjects $SUBJECTS
fi
if [ -n "$SESSIONS" ]; then
  set -- "$@" --sessions $SESSIONS
fi
if [ "$DRY_RUN" -eq 1 ]; then
  set -- "$@" --dry-run
fi
if [ "$GROUP_BY_SUBJ" -eq 1 ]; then
  set -- "$@" --group-by-subject
fi

# Run from project root to pick up local package
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

echo "Writing shards for derivatives: $DERIV_DIR"
if [ -n "$OUT_DIR" ]; then
  echo "Output dir: $OUT_DIR"
fi

if command -v uv >/dev/null 2>&1; then
  ( cd "$PROJ_DIR" && PYTHONPATH="$PROJ_DIR/src" uv run python -m structprep.make_wds "$@" )
else
  PYTHONPATH="$PROJ_DIR/src" python -m structprep.make_wds "$@"
fi

echo "Done."
