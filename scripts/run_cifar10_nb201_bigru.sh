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
# nap2 predictor: alonshn's CIFAR-10 model trained with the NAPv2 repo
# (log-norm AEs + BiGRU dual-path, embedding order [weights, gradients] =
# the repo default). Local copy of the cluster's
# /groups/giladkz_group/alonshn/NAPv2_assets/ae_lognorm_seed42/c10.
# Point NAP2_MODEL_DIR at any other model dir with the same ae/ + lstm/
# layout (e.g. ~/NAP2_models/pairs/c100_in16).
#
# normalize=log and predictor_type=bigru are read from the JSONs; the
# --nap2_lstm_* flag names predate BiGRU support but accept either type.
# ----------------------------------------------------------------------
MODEL_DIR="${NAP2_MODEL_DIR:-$HOME/NAP2_models/single/c10}"

AE_WEIGHTS_PT="$MODEL_DIR/ae/weights/best_ae_model.pt"
AE_WEIGHTS_JSON="$MODEL_DIR/ae/weights/model_hyper_params.json"
AE_GRADIENTS_PT="$MODEL_DIR/ae/gradients/best_ae_model.pt"
AE_GRADIENTS_JSON="$MODEL_DIR/ae/gradients/model_hyper_params.json"
PREDICTOR_PT="$MODEL_DIR/lstm/cp/model_state_cp/model.pt"
PREDICTOR_JSON="$MODEL_DIR/lstm/model_hyper_params.json"

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
