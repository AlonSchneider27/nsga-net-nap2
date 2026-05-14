#!/usr/bin/env bash
# Reproducible NB201 CIFAR-10 search using a BiGRU nap2 predictor.
# Seed: 42. Run scale: 40 pop x 30 gens x 20 epochs.
#
# Architecture format: NB201 canonical arch_str (5**6 = 15,625-arch design
# space). Every architecture the GA samples is a known NB201 catalog entry,
# so summary.json's `genotype` field can be grep'd directly against the
# NB201 paper's catalog dump.
#
# Re-run with: bash scripts/run_cifar10_nb201_bigru.sh
set -euo pipefail

# ----------------------------------------------------------------------
# nap2 predictor checkpoints (BiGRU + matching log-norm AEs).
#
# AEs live under .../log_transform/cifar10/...  -> the AEs were trained
# with --normalize log; our _load_nap2_predictor auto-detects this from
# the AE weights JSON, so no extra wiring is needed here.
#
# The predictor JSON must contain "predictor_type": "bigru" so the
# loader instantiates BiGRUDualPredictor (not LSTMPredictor). The
# --nap2_lstm_* flag names predate BiGRU support but accept either
# predictor type.
# ----------------------------------------------------------------------
AE_BASE=/sise/giladkz-group/Gilad-Group/michael/cross-dataset-autoresearch/controlled_aes/log_transform/cifar10
PREDICTOR_DIR=/sise/giladkz-group/Gilad-Group/michael/cross-dataset-autoresearch/lookup_tables/seed42
PREDICTOR_JSON_DIR=/sise/giladkz-group/Gilad-Group/alonshn/files_for_nap2_runs

AE_WEIGHTS_PT="$AE_BASE/weights/best_ae_model.pt"
AE_WEIGHTS_JSON="$AE_BASE/weights/model_hyper_params.json"
AE_GRADIENTS_PT="$AE_BASE/gradients/best_ae_model.pt"
AE_GRADIENTS_JSON="$AE_BASE/gradients/model_hyper_params.json"
PREDICTOR_PT="$PREDICTOR_DIR/model.pt"
PREDICTOR_JSON="$PREDICTOR_JSON_DIR/model_hyper_params.json"

# Activate the project venv if present (no-op when PATH already points
# at the right python, e.g. on a Slurm node with a module-loaded env).
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python search/evolution_search.py \
  --seed 42 \
  --search_space nb201 \
  --dataset cifar10 \
  --init_channels 16 \
  --layers 15 \
  --epochs 20 \
  --pop_size 40 \
  --n_offspring 20 \
  --n_gens 30 \
  --output_dir experiments/cifar10_nb201_bigru \
  --use_nap2 \
  --nap2_ae_weights_pt     "$AE_WEIGHTS_PT" \
  --nap2_ae_weights_json   "$AE_WEIGHTS_JSON" \
  --nap2_ae_gradients_pt   "$AE_GRADIENTS_PT" \
  --nap2_ae_gradients_json "$AE_GRADIENTS_JSON" \
  --nap2_lstm_pt           "$PREDICTOR_PT" \
  --nap2_lstm_json         "$PREDICTOR_JSON"
