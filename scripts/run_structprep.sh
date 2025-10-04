#!/usr/bin/env sh
set -eu

usage() {
  cat <<'USAGE'
Run structprep (structural MRI preprocessing) with sensible, high-utilization defaults.

Required:
  -i, --input-dir   Input BIDS-like directory (contains sub-*/)
  -o, --output-dir  Output derivatives directory

Optional:
  -s, --subjects    Quoted space-separated subject IDs (e.g. "sub-001 sub-002").
                    If omitted, subjects auto-discovered from input dir.
  -e, --sessions    Quoted space-separated session IDs (e.g. "ses-01 ses-02").
  -M, --modalities  Modalities (CSV). Default: T1w,T2w,FLAIR
  -a, --mask-agg    Mask aggressiveness: liberal|medium|conservative (default: liberal)
  -b, --fs-bin      Path to FreeSurfer bin (contains mri_coreg, mri_vol2vol) (default: /opt/freesurfer/bin)
  -n, --iso-mm      Isotropic voxel size (mm). Default: 1.0
  -p, --shape       In-plane shape HxW (default: 256x256)
      --keep-depth  Preserve depth (D); only crop/pad H and W (default)
      --no-keep-depth  Allow crop/pad of depth (D)
  -j, --n-jobs      Parallel jobs (default: auto; see below)
  -t, --omp         OMP threads per job (default: auto; see below)
  -d, --dry-run     Do not write outputs
  -h, --help        Show this help

Concurrency (maximum effect defaults):
  - Detects total CPU cores (C).
  - Auto subjects list length (S) if not provided.
  - Defaults: n_jobs = min(S, C), omp_nthreads = max(1, C / n_jobs), nprocs = C.
  - This saturates CPU across parallel jobs while avoiding oversubscription.

Examples:
  structprep/scripts/run_structprep.sh \
    -i /data/bids -o /data/derivatives/structprep \
    -a liberal

USAGE
}

# Defaults
INPUT_DIR="/home/mihirneal/ADNI-part1/part1"
OUTPUT_DIR="/data/ADNI_struct"
SUBJECTS=""
SESSIONS=""
MODALITIES="T1w,FLAIR"
MASK_AGG="liberal"
FS_BIN="/opt/freesurfer/bin"
ISO_MM="1.0"
SHAPE="256x256"
N_JOBS="auto"
OMP_THREADS="auto"
DRY_RUN=0
KEEP_DEPTH=1

# Ensure FreeSurfer environment is available in non-interactive shells
# Many users have it only in their interactive shell rc (e.g., .zshrc),
# but this script runs under /bin/sh and via subprocess (uv), which won't inherit it.
# Try to source SetUpFreeSurfer.sh if FREESURFER_HOME is set or at /opt/freesurfer.
if [ -z "${FREESURFER_HOME:-}" ]; then
  if [ -d "/opt/freesurfer" ] && [ -f "/opt/freesurfer/SetUpFreeSurfer.sh" ]; then
    export FREESURFER_HOME="/opt/freesurfer"
    # Temporarily disable nounset while sourcing FreeSurfer setup
    _orig_opts="$-"
    set +u
    . "/opt/freesurfer/SetUpFreeSurfer.sh"
    case "$_orig_opts" in *u*) set -u ;; esac
  fi
else
  if [ -f "$FREESURFER_HOME/SetUpFreeSurfer.sh" ]; then
    _orig_opts="$-"
    set +u
    . "$FREESURFER_HOME/SetUpFreeSurfer.sh"
    case "$_orig_opts" in *u*) set -u ;; esac
  fi
fi

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      -i|--input-dir) INPUT_DIR="$2"; shift 2;;
      -o|--output-dir) OUTPUT_DIR="$2"; shift 2;;
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
      -M|--modalities) MODALITIES="$2"; shift 2;;
      -a|--mask-agg) MASK_AGG="$2"; shift 2;;
      -b|--fs-bin) FS_BIN="$2"; shift 2;;
      -n|--iso-mm) ISO_MM="$2"; shift 2;;
      -p|--shape) SHAPE="$2"; shift 2;;
      --keep-depth) KEEP_DEPTH=1; shift 1;;
      --no-keep-depth) KEEP_DEPTH=0; shift 1;;
      -j|--n-jobs) N_JOBS="$2"; shift 2;;
      -t|--omp) OMP_THREADS="$2"; shift 2;;
      -d|--dry-run) DRY_RUN=1; shift 1;;
      -h|--help) usage; exit 0;;
      --) shift; break;;
      *) echo "Unknown argument: $1"; usage; exit 1;;
    esac
  done
}

parse_args "$@"

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "ERROR: --input-dir and --output-dir are required" >&2
  usage
  exit 1
fi

# Detect cores
detect_cores() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
  else
    sysctl -n hw.ncpu 2>/dev/null || echo 4
  fi
}

CORES=$(detect_cores)
NPROCS="$CORES"

SUBJECT_LIST=""
if [ -n "$SUBJECTS" ]; then
  SUBJECT_LIST="$SUBJECTS"
else
  for d in "$INPUT_DIR"/sub-*; do
    if [ -d "$d" ]; then
      base=$(basename "$d")
      if [ -n "$SUBJECT_LIST" ]; then SUBJECT_LIST="$SUBJECT_LIST $base"; else SUBJECT_LIST="$base"; fi
    fi
  done
fi

set -- $SUBJECT_LIST
SUBJECT_COUNT=$#

if [ "$SUBJECT_COUNT" -eq 0 ]; then
  echo "ERROR: No subjects found. Provide --subjects or ensure $INPUT_DIR/sub-* exists." >&2
  exit 1
fi

if [ "$N_JOBS" = "auto" ]; then
  if [ "$SUBJECT_COUNT" -gt "$CORES" ]; then
    N_JOBS=$CORES
  else
    N_JOBS=$SUBJECT_COUNT
  fi
fi

if [ "$OMP_THREADS" = "auto" ]; then
  if [ "$N_JOBS" -gt 0 ]; then
    OMP_THREADS=`expr "$CORES" / "$N_JOBS"`
  else
    OMP_THREADS="$CORES"
  fi
  if [ "$OMP_THREADS" -lt 1 ]; then OMP_THREADS=1; fi
fi

# If --fs-bin not set, try to infer from FREESURFER_HOME
if [ -z "$FS_BIN" ] && [ -n "${FREESURFER_HOME:-}" ] && [ -d "$FREESURFER_HOME/bin" ]; then
  FS_BIN="$FREESURFER_HOME/bin"
fi

# Preflight: let the user know how FS will be resolved
if [ -n "$FS_BIN" ] ; then
  if [ ! -x "$FS_BIN/mri_coreg" ] || [ ! -x "$FS_BIN/mri_vol2vol" ] || [ ! -x "$FS_BIN/mri_watershed" ] || [ ! -x "$FS_BIN/mri_binarize" ] || [ ! -x "$FS_BIN/lta_convert" ]; then
    echo "WARNING: --fs-bin '$FS_BIN' missing one or more: mri_coreg, mri_vol2vol, mri_watershed, mri_binarize, lta_convert" >&2
  fi
else
  if ! command -v mri_coreg >/dev/null 2>&1 || ! command -v mri_vol2vol >/dev/null 2>&1 || ! command -v mri_watershed >/dev/null 2>&1 || ! command -v mri_binarize >/dev/null 2>&1 || ! command -v lta_convert >/dev/null 2>&1; then
    echo "WARNING: FreeSurfer tools not on PATH and --fs-bin not set; missing one of mri_coreg, mri_vol2vol, mri_watershed, mri_binarize, lta_convert. Consider -b \"/path/to/freesurfer/bin\"." >&2
  fi
fi

echo "Detected cores: $CORES"
echo "Subjects: $SUBJECT_COUNT | n_jobs: $N_JOBS | omp_nthreads: $OMP_THREADS | nprocs: $NPROCS"
# Logs directory per subject
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

run_one_subject() {
  subj="$1"
  log_file="$2"
  # Build argument list
  set -- \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --subjects "$subj" \
    --modalities "$MODALITIES" \
    --isotropic "$ISO_MM" \
    --shape "$SHAPE" \
    $( [ "$KEEP_DEPTH" -eq 1 ] && echo --keep-depth ) \
    --mask-aggressiveness "$MASK_AGG" \
    --n-jobs 1 \
    --nprocs "$NPROCS" \
    --omp-nthreads "$OMP_THREADS" \
    --mem-mb 16000

  if [ -n "$SESSIONS" ]; then
    set -- "$@" --sessions $SESSIONS
  fi
  if [ -n "$FS_BIN" ]; then
    set -- "$@" --fs-bin "$FS_BIN"
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    set -- "$@" --dry-run
  fi

  # Run from project root and expose src on PYTHONPATH for local import
  SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
  PROJ_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
  ( cd "$PROJ_DIR" && PYTHONPATH="$PROJ_DIR/src" uv run python -m structprep run "$@" ) >> "$log_file" 2>&1

  # Determine success by scanning the log for any session failures
  FAIL_SES=""
  if grep '^✗ Failed ' "$log_file" >/dev/null 2>&1; then
    # Extract sessions (4th token, strip trailing colon)
    FAIL_SES=$(grep '^✗ Failed ' "$log_file" | awk '{sub(/:.*/,"",$4); print $4}' | tr '\n' ' ' | sed 's/[[:space:]]*$//')
    echo "$subj FAIL $FAIL_SES" > "$log_file.result"
    return 1
  else
    echo "$subj OK" > "$log_file.result"
    return 0
  fi
}

# Launch with subject-level parallelism up to N_JOBS (POSIX sh)
PIDS=""
SUBS_PENDING=""
LOGS_PENDING=""
SUCC=0
FAIL=0
FAILED_SUMMARY=""

for s in $SUBJECT_LIST; do
  ts=$(date +%Y%m%d-%H%M%S)
  log_file="$LOG_DIR/${s}_${ts}.log"
  echo "[$s] START"
  run_one_subject "$s" "$log_file" &
  pid=$!
  PIDS="$PIDS $pid"
  if [ -n "$SUBS_PENDING" ]; then SUBS_PENDING="$SUBS_PENDING $s"; else SUBS_PENDING="$s"; fi
  if [ -n "$LOGS_PENDING" ]; then LOGS_PENDING="$LOGS_PENDING $log_file"; else LOGS_PENDING="$log_file"; fi

  # Throttle by waiting for the first PID when at capacity
  set -- $PIDS
  while [ "$#" -ge "$N_JOBS" ]; do
    first_pid="$1"; shift; PIDS="$*"
    # Pop matching subject and log
    set -- $SUBS_PENDING; first_sub="$1"; shift; SUBS_PENDING="$*"
    set -- $LOGS_PENDING; first_log="$1"; shift; LOGS_PENDING="$*"

    wait "$first_pid"
    status=$?
    if [ "$status" -eq 0 ]; then
      echo "[$first_sub] OK"
      SUCC=`expr "$SUCC" + 1`
    else
      # Read failed sessions from result file if present
      FAIL_SES=""
      if [ -f "$first_log.result" ]; then
        # shellcheck disable=SC2034
        status_line=$(cat "$first_log.result")
        FAIL_SES=$(echo "$status_line" | awk '{for (i=3;i<=NF;i++) printf $i" ";}')
        FAIL_SES=$(echo "$FAIL_SES" | sed 's/[[:space:]]*$//')
      fi
      echo "[$first_sub] FAIL${FAIL_SES:+ (sessions: $FAIL_SES)}"
      FAIL=`expr "$FAIL" + 1`
      if [ -n "$FAILED_SUMMARY" ]; then FAILED_SUMMARY="$FAILED_SUMMARY, $first_sub${FAIL_SES:+[$FAIL_SES]}"; else FAILED_SUMMARY="$first_sub${FAIL_SES:+[$FAIL_SES]}"; fi
    fi
    set -- $PIDS
  done
done

# Wait for remaining
for pid in $PIDS; do
  # Pop corresponding subject/log
  set -- $SUBS_PENDING; first_sub="$1"; shift; SUBS_PENDING="$*"
  set -- $LOGS_PENDING; first_log="$1"; shift; LOGS_PENDING="$*"

  wait "$pid"
  status=$?
  if [ "$status" -eq 0 ]; then
    echo "[$first_sub] OK"
    SUCC=`expr "$SUCC" + 1`
  else
    FAIL_SES=""
    if [ -f "$first_log.result" ]; then
      status_line=$(cat "$first_log.result")
      FAIL_SES=$(echo "$status_line" | awk '{for (i=3;i<=NF;i++) printf $i" ";}')
      FAIL_SES=$(echo "$FAIL_SES" | sed 's/[[:space:]]*$//')
    fi
    echo "[$first_sub] FAIL${FAIL_SES:+ (sessions: $FAIL_SES)}"
    FAIL=`expr "$FAIL" + 1`
    if [ -n "$FAILED_SUMMARY" ]; then FAILED_SUMMARY="$FAILED_SUMMARY, $first_sub${FAIL_SES:+[$FAIL_SES]}"; else FAILED_SUMMARY="$first_sub${FAIL_SES:+[$FAIL_SES]}"; fi
  fi
done

echo "Summary: OK=$SUCC, FAIL=$FAIL"
if [ "$FAIL" -gt 0 ]; then
  echo "Failed subjects: $FAILED_SUMMARY"
fi
echo "Logs in: $LOG_DIR"
