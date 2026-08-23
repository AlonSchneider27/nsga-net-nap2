"""LC-PFN scorer (Adriaensen et al., arXiv:2310.20447) over the vendored lcpfn.

The pretrained transformer extrapolates a monotone-increasing val-acc curve in
[0,1]. It needs >=5 observations to be reliable (per the NAP2 paper's
protocol); below that, or on a NaN prediction, the score falls back to the
last observed val acc (the Early-Stop score).
"""

import logging
import os

import torch

from fitness.scorers import register_fitness

MIN_OBSERVATIONS = 5
# The lcpfn prior was trained on curves of length ~100; targets beyond that
# are out of distribution and get rescaled onto [0, 100].
MAX_IN_DIST_X = 100


@register_fitness('lc_pfn')
class LCPFNScorer:
    needs_val_curve = True
    needs_final_val = False

    def __init__(self, ckpt_path, target_epochs=20):
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise ValueError(
                f'lc_pfn: checkpoint not found at {ckpt_path!r} — run '
                'scripts/fetch_lcpfn_checkpoint.sh and pass its output path')
        self.ckpt_path = ckpt_path
        self.target_epochs = target_epochs
        self._model = None

    def _load(self):
        if self._model is None:
            import lcpfn   # vendored at repo root
            self._model = lcpfn.LCPFN(model_name=self.ckpt_path)
        return self._model

    def score(self, trace):
        curve = trace.val_acc_curve
        k = len(curve)
        if k < MIN_OBSERVATIONS:
            logging.warning('lc_pfn: only %d observations (<%d), falling back '
                            'to early-stop score', k, MIN_OBSERVATIONS)
            return float(curve[-1])

        x_target = float(max(round(self.target_epochs * trace.epoch_len
                                   / trace.snapshot_interval), k + 1))
        x = torch.arange(1, k + 1, dtype=torch.float32)
        if x_target > MAX_IN_DIST_X:
            logging.warning('lc_pfn: target x=%d beyond in-distribution range, '
                            'rescaling axis to [0, %d]', int(x_target), MAX_IN_DIST_X)
            x = x * (MAX_IN_DIST_X / x_target)
            x_target = float(MAX_IN_DIST_X)

        model = self._load()
        y = torch.tensor(curve, dtype=torch.float32).clamp(0.0, 1.0)
        prediction = model.predict_mean(
            x_train=x.unsqueeze(1),
            y_train=y.unsqueeze(1),
            x_test=torch.tensor([[x_target]], dtype=torch.float32),
        ).item()

        if prediction != prediction:   # NaN
            logging.warning('lc_pfn: NaN prediction, falling back to '
                            'early-stop score')
            return float(curve[-1])
        return float(prediction)
