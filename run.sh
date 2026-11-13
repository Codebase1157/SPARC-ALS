#!/usr/bin/env bash
# =============================================================================
# SPARC-ALS v1.0.0 — top-level driver
#
# Reproduces every table and figure reported in:
#   Ahuja & Kaur, "SPARC-ALS: Speaker-Invariant Pathology-Aware
#   Representation with Vowel-Space Graph Attention for Cross-Corpus
#   Detection of Amyotrophic Lateral Sclerosis", IJIES, 2026.
#
# Usage:
#   bash run.sh                 # all seven experiments, all seeds
#   bash run.sh E1              # only within-corpus LOSO
#   bash run.sh E1 E3 E5        # sub-set
#   bash run.sh --dry-run       # shape-only validation, no GPU
#
# All random state flows from --seeds (default: 11 23 42 77 2025), making the
# released per-speaker / per-utterance predictions exactly reproducible.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONHASHSEED=0

SEEDS=(11 23 42 77 2025)
EXPERIMENTS=(E1 E2 E3 E4 E5 E6 E7)
DRY_RUN=0

# Parse arguments
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    E1|E2|E3|E4|E5|E6|E7) ARGS+=("$arg") ;;
    --seeds=*) IFS=',' read -r -a SEEDS <<< "${arg#*=}" ;;
    -h|--help)
      head -n 20 "$0" | tail -n 19 | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown argument: $arg" >&2 ; exit 2 ;;
  esac
done
if [[ ${#ARGS[@]} -gt 0 ]]; then
  EXPERIMENTS=("${ARGS[@]}")
fi

DRY_FLAG=""
if [[ $DRY_RUN -eq 1 ]]; then DRY_FLAG="--dry-run"; fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "SPARC-ALS v1.0.0   reproduction driver"
echo "  project root : ${PROJECT_ROOT}"
echo "  seeds        : ${SEEDS[*]}"
echo "  experiments  : ${EXPERIMENTS[*]}"
echo "  dry run      : ${DRY_RUN}"
echo "========================================================================"

# ---------------------------------------------------------------------------
# Stage 0: Preprocessing + reference statistics
#   (idempotent — skipped if data/features is present and non-empty)
# ---------------------------------------------------------------------------
if [[ ! -d data/features || -z "$(ls -A data/features 2>/dev/null || true)" ]]; then
  echo
  echo "--- Stage 0: preprocessing + MFA forced alignment + reference stats"
  python -m sparc_als.data.preprocess        --config configs/preprocessing.yaml   $DRY_FLAG
  python -m sparc_als.data.forced_alignment  --config configs/forced_alignment.yaml $DRY_FLAG
  python -m sparc_als.utils.reference_stats  --config configs/acoustic_targets.yaml $DRY_FLAG
fi

# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------
for exp in "${EXPERIMENTS[@]}"; do
  echo
  echo "========================================================================"
  echo "Running ${exp}"
  echo "========================================================================"
  for seed in "${SEEDS[@]}"; do
    echo "  seed = ${seed}"
    python "scripts/${exp}_"*.py --seed "${seed}" $DRY_FLAG
  done
done

# ---------------------------------------------------------------------------
# Aggregation + paired tests on seed-averaged speaker scores
# ---------------------------------------------------------------------------
echo
echo "--- Aggregating seed-averaged predictions and running paired tests"
python -m sparc_als.eval.aggregate     --seeds "${SEEDS[@]}" $DRY_FLAG
python -m sparc_als.eval.paired_tests  --seeds "${SEEDS[@]}" $DRY_FLAG

echo
echo "Done. See predictions/ for per-speaker / per-utterance logits tagged by"
echo "seed_id, and results/ for aggregated tables and figures."
