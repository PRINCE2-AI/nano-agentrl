#!/usr/bin/env bash
# Ablations on top of one base config (default: the 0.5B search config).
#
#   bash scripts/run_ablations.sh                         # all ablations, seed 42
#   bash scripts/run_ablations.sh configs/search_1.5b.yaml "1 2 3"   # 3 seeds
#   ONLY="nomask drgrpo" bash scripts/run_ablations.sh    # a subset
#   DRY_RUN=1 bash scripts/run_ablations.sh               # tiny model, CPU, seconds
#
# Every run logs to W&B group "ablations" and writes runs/ablations/<name>_s<seed>/.
# Afterwards: scripts/reliability_sweep.py on each run dir, then scripts/plot.py with
# --metrics runs/ablations/*/metrics.jsonl to overlay the curves.
#
#   name      change                                       question it answers
#   baseline  none                                         reference curve
#   nomask    mask_retrieved_tokens=false                  does masking <information> matter?
#   drgrpo    loss_agg=dr_grpo, normalize_std=false        GRPO vs Dr. GRPO
#   g4        group_size=4, prompts_per_step x2            G=4 vs G=8 at equal trajectories/step
#   base      base model (no -Instruct), no chat template  base vs instruct start
set -euo pipefail

CONFIG="${1:-configs/search_0.5b.yaml}"
SEEDS="${2:-42}"
ONLY="${ONLY:-baseline nomask drgrpo g4 base}"
EXTRA=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then EXTRA+=(--dry_run); fi

# Read the base config's model and prompts_per_step so g4 / base derive from it.
info=$(python -c "
import sys
from nano_agentrl.config import load_config
c = load_config(sys.argv[1])
print(c.model_name, c.prompts_per_step)
" "$CONFIG" | tr -d '\r')   # Windows Python prints CRLF
read -r MODEL PPS <<< "$info"
BASE_MODEL="${MODEL%-Instruct}"

ablation_args() {
  case "$1" in
    baseline) echo "" ;;
    nomask)   echo "--set mask_retrieved_tokens=false" ;;
    drgrpo)   echo "--set loss_agg=dr_grpo --set normalize_std=false" ;;
    g4)       echo "--set group_size=4 --set prompts_per_step=$((PPS * 2))" ;;
    base)     echo "--set model_name=${BASE_MODEL} --set use_chat_template=false" ;;
    *)        echo "unknown ablation: $1" >&2; return 1 ;;
  esac
}

for seed in $SEEDS; do
  for name in $ONLY; do
    run="runs/ablations/${name}_s${seed}"
    args=$(ablation_args "$name")   # a plain assignment, so set -e stops on an unknown name
    echo "=== ${name} (seed ${seed}) -> ${run}  ${args}"
    # shellcheck disable=SC2086  # word-splitting of the --set flags is intended
    python scripts/train.py --config "$CONFIG" --run_dir "$run" \
      --set seed="$seed" --set wandb_group=ablations --set wandb_run_name="${name}_s${seed}" \
      $args "${EXTRA[@]}"
  done
done
