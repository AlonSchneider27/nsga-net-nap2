"""Minimal example: predict an architecture's accuracy with NAP2.

NAP2 predicts a network's *final* accuracy from a few snapshots taken during
the first steps of training -- no need to train to convergence. You provide
(1) a model and (2) a training dataloader; NAP2 partially trains the model,
encodes weight/gradient snapshots through two autoencoders, and runs a
sequence predictor (LSTM or BiGRU) to output a predicted accuracy in [0, 1].

Run from the project root (so the `nap2` package is importable):

    python examples/nap2_inference.py
"""

import json

import torch  # noqa: F401  (NAP2 needs torch; also handy if you build a model below)
from torch.utils.data import DataLoader

from nap2.predictor import NAP2Predictor
from nap2.autoencoder import FeatureMapAutoEncoder
from nap2.bigru_predictor import BiGRUDualPredictor
# If your predictor is an LSTM rather than a BiGRU, swap the import:
# from nap2.lstm_predictor import LSTMPredictor

# ---------------------------------------------------------------------------
# 1. Checkpoint paths -- EDIT THESE.
#    Three trained pieces, each a .pt (weights) + .json (hyperparameters):
#      - weights  autoencoder
#      - gradients autoencoder
#      - sequence predictor (LSTM or BiGRU)
# ---------------------------------------------------------------------------
AE_WEIGHTS_PT     = "/path/to/ae/weights/best_ae_model.pt"
AE_WEIGHTS_JSON   = "/path/to/ae/weights/model_hyper_params.json"
AE_GRADIENTS_PT   = "/path/to/ae/gradients/best_ae_model.pt"
AE_GRADIENTS_JSON = "/path/to/ae/gradients/model_hyper_params.json"
PREDICTOR_PT      = "/path/to/predictor/model.pt"
PREDICTOR_JSON    = "/path/to/predictor/model_hyper_params.json"

# ---------------------------------------------------------------------------
# 2. Load the two autoencoders + the predictor, then assemble the pipeline.
# ---------------------------------------------------------------------------
ae_weights   = FeatureMapAutoEncoder.load(model_path=AE_WEIGHTS_PT,   params_path=AE_WEIGHTS_JSON)
ae_gradients = FeatureMapAutoEncoder.load(model_path=AE_GRADIENTS_PT, params_path=AE_GRADIENTS_JSON)
predictor_net = BiGRUDualPredictor.load(model_path=PREDICTOR_PT, params_path=PREDICTOR_JSON)

# `normalize` must match how the AEs were trained; it's recorded in the AE JSON.
with open(AE_WEIGHTS_JSON) as f:
    normalize = json.load(f).get("normalize", "none")

predictor = NAP2Predictor(
    ae_weights=ae_weights,
    ae_gradients=ae_gradients,
    lstm=predictor_net,   # the kwarg is named `lstm` but accepts an LSTM *or* BiGRU
    normalize=normalize,
)

# (Shortcut: if your files follow the expected on-disk layout, you can replace
#  steps 1-2 with a single call -- NAP2Predictor.load("/path/to/model_dir").)

# ---------------------------------------------------------------------------
# 3. The architecture to score + a training dataloader.
#    `model` is any nn.Module that outputs `num_classes` logits.
#    `dataloader` yields (input, target) batches matching that model.
# ---------------------------------------------------------------------------
model = ...                       # e.g. torchvision.models.resnet18(num_classes=10)
dataloader = DataLoader(...)      # e.g. a CIFAR-10 training loader, batch_size=256

# ---------------------------------------------------------------------------
# 4. Inference. `steps` = number of snapshots collected during partial training.
#    Returns the predicted final accuracy as a float in [0, 1].
# ---------------------------------------------------------------------------
pred_acc = predictor.score(model, dataloader, steps=5)
print(f"NAP2 predicted accuracy: {pred_acc:.4f}")
