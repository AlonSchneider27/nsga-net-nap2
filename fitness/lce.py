"""LCE-m: parametric learning-curve extrapolation (Domhan et al., IJCAI 2015).

Port of NASLib's ``naslib/predictors/lce/`` (itself from
github.com/automl/pylearningcurvepredictor, author Tobias Domhan), the code
benchmarked in White et al. 2021 (arXiv:2104.01177) and the LCE-m baseline of
the NAP2 paper. Kept verbatim: the pruned 4-family list, per-family MLE via
L-BFGS-B on the Gaussian NLL (all-ones init, trailing parameter = sigma),
equal-weight ensemble, random-walk Metropolis (N=300 for NB201, proposal
scale 1e-4, monotonicity accept filter), posterior-mean prediction, and the
default-guess-plus-jitter fallback on NaN.

One deliberate deviation: NASLib's ``perturb_params`` shallow-copies the
params dict and then ``+=``-mutates the shared numpy arrays in place, so
model-parameter perturbations are applied even when the Metropolis test
rejects, and every recorded sample aliases the same arrays. We copy the
arrays per candidate so rejection actually rejects, which is the algorithm
Domhan describes.
"""

import logging

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from fitness.scorers import register_fitness


def _logloglinear(params, x):
    a, b, _ = params
    return np.log(a * np.log(x) + b)


def _logpower(params, x):
    a, b, c, _ = params
    return a / (1 + (x / np.exp(b)) ** c)


def _mmf(params, x):
    alpha, beta, delta, kappa, _ = params
    return alpha - (alpha - beta) / (1 + (kappa * x) ** delta)


def _hill3(params, x):
    # NASLib formula, ported as-is (kappa * eta in the denominator).
    ymax, eta, kappa, _ = params
    return ymax * (x ** eta) / (kappa * eta + x ** eta)


_positive_only = (np.finfo(float).eps, np.inf)
_no_bound = (-np.inf, np.inf)

# name -> (fn, degrees of freedom, L-BFGS-B bounds incl. trailing sigma slot)
MODEL_CONFIG = {
    "logloglinear": (_logloglinear, 2, (_positive_only,) * 3),
    "logpower": (_logpower, 3, None),
    "mmf": (_mmf, 4, None),
    "hill3": (_hill3, 3, (_positive_only, _no_bound, _positive_only, _positive_only)),
}


class _ParametricModel:
    def __init__(self, name):
        self.fn, self.degrees_freedom, self.bounds = MODEL_CONFIG[name]
        self.name = name

    def fit(self, y):
        x = np.arange(1, y.shape[0] + 1, dtype=float)

        def nll(parameters):
            residuals = y - self.predict(x, parameters)
            return (len(x) / 2 * np.log(2 * np.pi)
                    + len(x) / 2 * np.log(parameters[-1] ** 2)
                    + 1 / (2 * parameters[-1] ** 2) * np.sum(residuals ** 2))

        opt = minimize(nll, np.ones(self.degrees_freedom + 1),
                       method="L-BFGS-B", bounds=self.bounds)
        self.params = opt["x"]

    def predict(self, x, params=None):
        with np.errstate(all="ignore"):
            return self.fn(self.params if params is None else params,
                           np.asarray(x, dtype=float))


class ParametricEnsemble:
    def __init__(self, model_names=tuple(MODEL_CONFIG)):
        self.models = [_ParametricModel(n) for n in model_names]
        self.weights = [1 / len(self.models)] * len(self.models)

    def fit(self, y):
        for model in self.models:
            model.fit(y)
        self.params = {m.name: m.params for m in self.models}
        preds = self.predict(np.arange(1, y.shape[0] + 1))
        self.sigma_sq = float(np.mean((y - preds) ** 2))

    def predict(self, x, params=None, weights=None):
        if params is None:
            params, weights = self.params, self.weights
        return sum(w * m.predict(x, params=params[m.name])
                   for w, m in zip(weights, self.models))

    def _log_likelihood(self, y, params, weights, sigma_sq):
        """(log-likelihood, min point likelihood) over the observed curve."""
        total = 0.0
        min_pl = 1.0
        for j in range(y.shape[0]):
            err = self.predict(j + 1, params=params, weights=weights) - y[j]
            with np.errstate(all="ignore"):
                pl = norm.pdf(err, scale=np.sqrt(sigma_sq))
            min_pl = min(min_pl, pl)
            if not pl > 0:      # also catches NaN
                pl = 1e-10
            total += np.log(pl)
        return total, min_pl

    def mcmc(self, y, N=300, var=0.0001):
        self.fit(y)
        curvelen = y.shape[0]
        params = {k: v.copy() for k, v in self.params.items()}
        weights = list(self.weights)
        sigma_sq = self.sigma_sq
        self.mcmc_samples = []

        n_free = 1 + len(weights) + sum(m.degrees_freedom for m in self.models)
        for _ in range(N):
            self.mcmc_samples.append((params, weights))

            current_ll, _ = self._log_likelihood(y, params, weights, sigma_sq)

            perturbation = np.random.normal(loc=0, scale=var, size=(n_free,))
            cand_params = {}
            pos = 0
            for m in self.models:
                delta = np.concatenate(
                    [perturbation[pos:pos + m.degrees_freedom], np.zeros(1)])
                cand_params[m.name] = params[m.name] + delta
                pos += m.degrees_freedom
            cand_weights = [w + perturbation[pos + i] for i, w in enumerate(weights)]
            cand_sigma_sq = sigma_sq + perturbation[-1]

            if cand_sigma_sq <= 0:
                continue
            cand_ll, min_pl = self._log_likelihood(y, cand_params, cand_weights,
                                                   cand_sigma_sq)
            if min_pl == 0:
                continue

            with np.errstate(over="ignore"):
                acceptance = min(1, np.exp(cand_ll - current_ll))
            # Monotonicity filter: only accept states whose curve increases.
            if self.predict(curvelen + 1, params=cand_params, weights=cand_weights) \
                    > self.predict(1, params=cand_params, weights=cand_weights):
                if np.random.random() < acceptance:
                    params, weights, sigma_sq = cand_params, cand_weights, cand_sigma_sq

    def mcmc_sample_predict(self, x):
        return sum(self.predict(x, params=p, weights=w)
                   for p, w in self.mcmc_samples) / len(self.mcmc_samples)


@register_fitness('lce_m')
class LCEMScorer:
    """Extrapolates the per-snapshot val-acc curve to the target horizon."""
    needs_val_curve = True
    needs_final_val = False

    def __init__(self, target_epochs=20, mcmc_steps=300):
        self.target_epochs = target_epochs
        self.mcmc_steps = mcmc_steps

    def score(self, trace):
        y = np.asarray(trace.val_acc_curve, dtype=float)
        # Snapshot index is the time unit; the target epoch count converts to
        # snapshot units through the runtime epoch length.
        x_target = max(round(self.target_epochs * trace.epoch_len
                             / trace.snapshot_interval), y.shape[0] + 1)
        ensemble = ParametricEnsemble()
        try:
            ensemble.mcmc(y, N=self.mcmc_steps)
            prediction = float(np.squeeze(ensemble.mcmc_sample_predict(x_target)))
        except Exception:
            logging.exception('lce_m: MCMC failed, using fallback')
            prediction = float('nan')
        if not np.isfinite(prediction):
            # NASLib's fallback (85.0 + U(0,1) on the 0-100 scale): a constant
            # default guess with jitter purely to break rank ties.
            prediction = 0.85 + np.random.rand() / 100
        return prediction
