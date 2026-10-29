"""The two 128-dim AE embeddings must be concatenated in the order the predictor
checkpoint was trained with: "wg" = [weights, gradients] (the NAPv2 repo's order,
the default) or "gw" = [gradients, weights] (michael's deployed checkpoints only).
A wrong order raises no error at inference -- predictions just collapse -- so the
resolution rules and the actual layout of the 256-vector are pinned here.
"""
import json

import numpy as np
import pytest
import torch

from nap2.autoencoder import FeatureMapAutoEncoder
from nap2.bigru_predictor import BiGRUDualPredictor
from nap2.predictor import NAP2Predictor, resolve_embedding_order


def _two_aes():
    torch.manual_seed(1)
    ae_w = FeatureMapAutoEncoder().eval()
    torch.manual_seed(2)
    ae_g = FeatureMapAutoEncoder().eval()
    return ae_w, ae_g


def _maps(n=2):
    rng = np.random.default_rng(0)
    return rng.random((n, 65, 100, 12)), rng.random((n, 65, 100, 12))


def _encode(ae, one_map):
    x = torch.tensor(np.moveaxis(one_map, -1, 0), dtype=torch.float64).unsqueeze(0)
    with torch.no_grad():
        return ae.encode(x).squeeze(0)


def test_default_order_is_weights_then_gradients():
    ae_w, ae_g = _two_aes()
    w, g = _maps()
    p = NAP2Predictor(ae_weights=ae_w, ae_gradients=ae_g, lstm=BiGRUDualPredictor())
    assert p._embedding_order == "wg"
    emb = p._encode_maps(w, g)
    assert emb.shape == (2, 256)
    assert torch.allclose(emb[0, :128], _encode(ae_w, w[0]))
    assert torch.allclose(emb[0, 128:], _encode(ae_g, g[0]))


def test_gw_order_swaps_the_halves():
    ae_w, ae_g = _two_aes()
    w, g = _maps()
    wg = NAP2Predictor(ae_weights=ae_w, ae_gradients=ae_g, lstm=BiGRUDualPredictor(),
                       embedding_order="wg")._encode_maps(w, g)
    gw = NAP2Predictor(ae_weights=ae_w, ae_gradients=ae_g, lstm=BiGRUDualPredictor(),
                       embedding_order="gw")._encode_maps(w, g)
    assert torch.equal(gw[:, :128], wg[:, 128:])
    assert torch.equal(gw[:, 128:], wg[:, :128])


def test_invalid_order_is_rejected():
    with pytest.raises(ValueError):
        NAP2Predictor(lstm=BiGRUDualPredictor(), embedding_order="weights_first")
    with pytest.raises(ValueError):
        resolve_embedding_order({"embedding_order": "xy"})


def test_resolve_from_predictor_json():
    assert resolve_embedding_order(None)[0] == "wg"
    assert resolve_embedding_order({"predictor_type": "bigru"})[0] == "wg"
    assert resolve_embedding_order({"embedding_order": "gw"}) == ("gw", "predictor_json:embedding_order")


# Exactly what NAPv2's train_lstm.py writes for a biGRU run (LSTM-only keys are null).
BIGRU_JSON = {
    "predictor_type": "bigru", "embedding_size": 256, "hidden_size": 128, "lstm_layers": None,
    "gru_layers": 2, "inner_dense_layer_sizes": None, "dense_size": 128, "dropout": 0.1,
    "last_layer": "sigmoid", "bi_directional": True, "batch_first": True, "is_double": True,
    "normalize": "log", "sharpness_features": None, "sharpness_only": False,
    "sharpness_norm": None, "aug": 30,
}


def _write_model_dir(root, predictor_json):
    for dt in ("weights", "gradients"):
        d = root / "ae" / dt
        d.mkdir(parents=True)
        torch.save(FeatureMapAutoEncoder().state_dict(), d / "best_ae_model.pt")
        json.dump({"layers_shapes": [[[1, 100], 512], [[65, 1], 256], [[1, 1], 128]],
                   "input_channels": 12, "normalize": "log"},
                  open(d / "model_hyper_params.json", "w"))
    (root / "lstm" / "cp" / "model_state_cp").mkdir(parents=True)
    torch.save(BiGRUDualPredictor().state_dict(), root / "lstm" / "cp" / "model_state_cp" / "model.pt")
    json.dump(predictor_json, open(root / "lstm" / "model_hyper_params.json", "w"))


def test_load_accepts_repo_trained_bigru_json_and_defaults_to_wg(tmp_path):
    # The null LSTM-only keys used to crash BiGRUDualPredictor.load (eager fallback indexing None).
    _write_model_dir(tmp_path, BIGRU_JSON)
    p = NAP2Predictor.load(str(tmp_path))
    assert type(p._lstm).__name__ == "BiGRUDualPredictor"
    assert p._normalize == "log"
    assert p._embedding_order == "wg"


def test_load_honours_gw_in_predictor_json(tmp_path):
    _write_model_dir(tmp_path, {**BIGRU_JSON, "embedding_order": "gw"})
    assert NAP2Predictor.load(str(tmp_path))._embedding_order == "gw"
