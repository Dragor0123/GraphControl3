#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASELINE_FILE="${BASELINE_FILE:-baseline_acc.txt}"
RESULT_DIR="${RESULT_DIR:-analysis_results/lambda_sweep}"
PHASE="${PHASE:-coarse}"
EPOCHS="${EPOCHS:-200}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SEEDS="${SEEDS:-0 1 2 3 4}"
DATASETS=()

usage() {
  cat <<'EOF'
Usage:
  bash command/sweep_lambda_gain.sh [options]

Options:
  --phase coarse|pilot
  --datasets "Cora_ML DBLP Photo Chameleon Squirrel Actor"
  --epochs 200
  --seeds "0 1 2 3 4"
  --cuda 0
  --baseline baseline_acc.txt
  --result-dir analysis_results/lambda_sweep

Notes:
  - `pilot` uses a smaller candidate set for faster first-pass exploration.
  - Results are appended to a timestamped CSV and summarized into a best-lambda CSV.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      PHASE="$2"
      shift 2
      ;;
    --datasets)
      read -r -a DATASETS <<< "$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --cuda)
      CUDA_DEVICE="$2"
      shift 2
      ;;
    --baseline)
      BASELINE_FILE="$2"
      shift 2
      ;;
    --result-dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$BASELINE_FILE" ]]; then
  echo "Baseline file not found: $BASELINE_FILE" >&2
  exit 1
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=(Cora_ML DBLP Photo Chameleon Squirrel Actor)
fi

mkdir -p "$RESULT_DIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_CSV="$RESULT_DIR/lambda_gain_sweep_${PHASE}_${TIMESTAMP}.csv"
BEST_CSV="$RESULT_DIR/lambda_gain_best_${PHASE}_${TIMESTAMP}.csv"

echo "dataset,lambda_gain,acc_mean,acc_std,baseline_mean,baseline_std,baseline_floor,acc_delta,meets_baseline,frozen_acc_mean,frozen_acc_std,control_acc_mean,control_acc_std,combined_acc_mean,combined_acc_std,mean_margin_gain_mean,mean_margin_gain_std,test_l_gain_mean,test_l_gain_std,train_l_gain_mean,train_l_gain_std,epochs,seeds" > "$RESULT_CSV"

baseline_value() {
  local dataset="$1"
  local column="$2"
  awk -F, -v d="$dataset" -v c="$column" '
    BEGIN { OFS="," }
    $1 == d {
      if (c == "mean") print $2;
      if (c == "std") print $3;
    }
  ' "$BASELINE_FILE"
}

threshold_for_dataset() {
  case "$1" in
    Chameleon|Squirrel) echo "0.15" ;;
    *) echo "0.17" ;;
  esac
}

lambda_grid_for_dataset() {
  local dataset="$1"
  local phase="$2"

  if [[ "$phase" == "pilot" ]]; then
    case "$dataset" in
      Cora_ML|DBLP|Photo) echo "0.02 0.10 0.30" ;;
      Chameleon) echo "0.10 0.50 1.00" ;;
      Squirrel|Actor) echo "0.25 1.00 2.00" ;;
      *) echo "0.10 0.50 1.00" ;;
    esac
    return
  fi

  case "$dataset" in
    Cora_ML|DBLP|Photo) echo "0.01 0.05 0.10 0.25 0.50" ;;
    Chameleon) echo "0.05 0.10 0.25 0.50 1.00 2.00" ;;
    Squirrel|Actor) echo "0.10 0.25 0.50 1.00 2.00 4.00" ;;
    *) echo "0.10 0.25 0.50 1.00" ;;
  esac
}

run_one() {
  local dataset="$1"
  local lambda_gain="$2"
  local threshold="$3"
  local baseline_mean="$4"
  local baseline_std="$5"
  local baseline_floor="$6"

  local safe_lambda
  safe_lambda="${lambda_gain//./p}"
  local log_file="$RESULT_DIR/${dataset}_lambda${safe_lambda}_${PHASE}_${TIMESTAMP}.log"

  echo
  echo "======================================================================"
  echo "Dataset: $dataset | lambda_gain=$lambda_gain | threshold=$threshold"
  echo "Baseline: mean=$baseline_mean std=$baseline_std floor=$baseline_floor"
  echo "Log: $log_file"
  echo "======================================================================"

  local run_output
  run_output="$(
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python graphcontrol.py \
      --model GCC_GraphControl \
      --dataset "$dataset" \
      --lr 0.5 \
      --optimizer adamw \
      --threshold "$threshold" \
      --use_adj \
      --epochs "$EPOCHS" \
      --walk_steps 256 \
      --restart 0.8 \
      --seeds $SEEDS \
      --use_l_gain \
      --lambda_gain "$lambda_gain" 2>&1 | tee "$log_file"
  )"

  parse_metric() {
    local metric_name="$1"
    local line
    line="$(printf '%s\n' "$run_output" | grep "# ${metric_name}:" | tail -n 1 || true)"
    if [[ -z "$line" ]]; then
      echo "Failed to parse ${metric_name} for dataset=$dataset lambda=$lambda_gain" >&2
      exit 1
    fi
    local mean std
    mean="$(printf '%s\n' "$line" | sed -E 's/.*: ([0-9.-]+)±([0-9.-]+)/\1/')"
    std="$(printf '%s\n' "$line" | sed -E 's/.*: ([0-9.-]+)±([0-9.-]+)/\2/')"
    printf '%s,%s\n' "$mean" "$std"
  }

  local acc_mean acc_std
  IFS=, read -r acc_mean acc_std <<< "$(parse_metric final_acc)"
  local frozen_acc_mean frozen_acc_std
  IFS=, read -r frozen_acc_mean frozen_acc_std <<< "$(parse_metric final_frozen_acc)"
  local control_acc_mean control_acc_std
  IFS=, read -r control_acc_mean control_acc_std <<< "$(parse_metric final_control_acc)"
  local combined_acc_mean combined_acc_std
  IFS=, read -r combined_acc_mean combined_acc_std <<< "$(parse_metric final_combined_acc)"
  local mean_margin_gain_mean mean_margin_gain_std
  IFS=, read -r mean_margin_gain_mean mean_margin_gain_std <<< "$(parse_metric final_mean_margin_gain)"
  local test_l_gain_mean test_l_gain_std
  IFS=, read -r test_l_gain_mean test_l_gain_std <<< "$(parse_metric final_test_l_gain)"
  local train_l_gain_mean train_l_gain_std
  IFS=, read -r train_l_gain_mean train_l_gain_std <<< "$(parse_metric final_train_l_gain)"

  local acc_delta meets_baseline
  acc_delta="$(awk -v a="$acc_mean" -v b="$baseline_mean" 'BEGIN { printf "%.4f", a - b }')"
  meets_baseline="$(awk -v a="$acc_mean" -v f="$baseline_floor" 'BEGIN { print (a >= f ? 1 : 0) }')"

  echo "$dataset,$lambda_gain,$acc_mean,$acc_std,$baseline_mean,$baseline_std,$baseline_floor,$acc_delta,$meets_baseline,$frozen_acc_mean,$frozen_acc_std,$control_acc_mean,$control_acc_std,$combined_acc_mean,$combined_acc_std,$mean_margin_gain_mean,$mean_margin_gain_std,$test_l_gain_mean,$test_l_gain_std,$train_l_gain_mean,$train_l_gain_std,$EPOCHS,\"$SEEDS\"" >> "$RESULT_CSV"
}

for dataset in "${DATASETS[@]}"; do
  baseline_mean="$(baseline_value "$dataset" mean)"
  baseline_std="$(baseline_value "$dataset" std)"
  if [[ -z "$baseline_mean" || -z "$baseline_std" ]]; then
    echo "Missing baseline row for dataset=$dataset in $BASELINE_FILE" >&2
    exit 1
  fi
  baseline_floor="$(awk -v m="$baseline_mean" -v s="$baseline_std" 'BEGIN { printf "%.4f", m - s }')"
  threshold="$(threshold_for_dataset "$dataset")"
  grid="$(lambda_grid_for_dataset "$dataset" "$PHASE")"

  for lambda_gain in $grid; do
    run_one "$dataset" "$lambda_gain" "$threshold" "$baseline_mean" "$baseline_std" "$baseline_floor"
  done
done

{
  printf "%s\n" "dataset,best_lambda_gain,acc_mean,acc_std,baseline_mean,baseline_std,baseline_floor,acc_delta,meets_baseline,frozen_acc_mean,frozen_acc_std,control_acc_mean,control_acc_std,combined_acc_mean,combined_acc_std,mean_margin_gain_mean,mean_margin_gain_std,test_l_gain_mean,test_l_gain_std,train_l_gain_mean,train_l_gain_std"
  awk -F, '
  NR == 1 { next }
  {
    dataset = $1
    lambda = $2 + 0
    acc = $3 + 0
    acc_std = $4 + 0
    base = $5 + 0
    base_std = $6 + 0
    floor = $7 + 0
    delta = $8 + 0
    meets = $9 + 0
    frozen_acc = $10 + 0
    frozen_acc_std = $11 + 0
    control_acc = $12 + 0
    control_acc_std = $13 + 0
    combined_acc = $14 + 0
    combined_acc_std = $15 + 0
    mean_margin_gain = $16 + 0
    mean_margin_gain_std = $17 + 0
    test_l_gain = $18 + 0
    test_l_gain_std = $19 + 0
    train_l_gain = $20 + 0
    train_l_gain_std = $21 + 0

    if (!(dataset in seen) ||
        meets > best_meets[dataset] ||
        (meets == best_meets[dataset] && mean_margin_gain > best_margin_gain[dataset]) ||
        (meets == best_meets[dataset] && mean_margin_gain == best_margin_gain[dataset] && acc > best_acc[dataset]) ||
        (meets == best_meets[dataset] && mean_margin_gain == best_margin_gain[dataset] && acc == best_acc[dataset] && lambda < best_lambda[dataset])) {
      seen[dataset] = 1
      best_lambda[dataset] = lambda
      best_acc[dataset] = acc
      best_acc_std[dataset] = acc_std
      best_base[dataset] = base
      best_base_std[dataset] = base_std
      best_floor[dataset] = floor
      best_delta[dataset] = delta
      best_meets[dataset] = meets
      best_frozen_acc[dataset] = frozen_acc
      best_frozen_acc_std[dataset] = frozen_acc_std
      best_control_acc[dataset] = control_acc
      best_control_acc_std[dataset] = control_acc_std
      best_combined_acc[dataset] = combined_acc
      best_combined_acc_std[dataset] = combined_acc_std
      best_margin_gain[dataset] = mean_margin_gain
      best_margin_gain_std[dataset] = mean_margin_gain_std
      best_test_l_gain[dataset] = test_l_gain
      best_test_l_gain_std[dataset] = test_l_gain_std
      best_train_l_gain[dataset] = train_l_gain
      best_train_l_gain_std[dataset] = train_l_gain_std
    }
  }
  END {
    for (dataset in seen) {
      printf "%s,%.6g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
        dataset,
        best_lambda[dataset],
        best_acc[dataset],
        best_acc_std[dataset],
        best_base[dataset],
        best_base_std[dataset],
        best_floor[dataset],
        best_delta[dataset],
        best_meets[dataset],
        best_frozen_acc[dataset],
        best_frozen_acc_std[dataset],
        best_control_acc[dataset],
        best_control_acc_std[dataset],
        best_combined_acc[dataset],
        best_combined_acc_std[dataset],
        best_margin_gain[dataset],
        best_margin_gain_std[dataset],
        best_test_l_gain[dataset],
        best_test_l_gain_std[dataset],
        best_train_l_gain[dataset],
        best_train_l_gain_std[dataset]
    }
  }
  ' "$RESULT_CSV" | sort
} > "$BEST_CSV"

echo
echo "Sweep results saved to: $RESULT_CSV"
echo "Best-lambda summary saved to: $BEST_CSV"
