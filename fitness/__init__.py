"""Learning-curve fitness baselines (NAP2 paper Sec 4.2, Tables 4-6).

Build scorers with ``build_scorers('sotl_e,lce_m', ...)`` or
``build_scorers('all', ...)``; each scorer maps a ``TrainingTrace`` (from
``fitness.trace.run_partial_train``) to a scalar where higher = better.
'nap2' is not part of this registry — it keeps its existing predictor path
in search/evolution_search.py.
"""

from fitness.scorers import FITNESS_SCORERS, register_fitness  # noqa: F401
from fitness.trace import TrainingTrace, run_partial_train  # noqa: F401

# Import implementations to trigger registration.
import fitness.lce  # noqa: F401
import fitness.lcpfn_scorer  # noqa: F401

# Deterministic 'all' order, matching the paper's table columns.
ALL_BASELINES = ['sotl', 'sotl_e', 'early_stop', 'lce_m', 'lc_pfn']


def build_scorers(spec, lcpfn_ckpt='', target_epochs=20):
    """Instantiate scorers from a comma-separated name list or 'all'."""
    if spec == 'all':
        names = list(ALL_BASELINES)
    else:
        names = [s.strip() for s in spec.split(',') if s.strip()]
    seen = set()
    scorers = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        if name == 'nap2':
            raise ValueError("'nap2' is handled by the caller via --use_nap2 / "
                             "the predictor path, not build_scorers()")
        if name not in FITNESS_SCORERS:
            raise ValueError(f'Unknown fitness method {name!r}. '
                             f'Available: {sorted(FITNESS_SCORERS)}')
        cls = FITNESS_SCORERS[name]
        if name == 'lc_pfn':
            scorers.append(cls(ckpt_path=lcpfn_ckpt, target_epochs=target_epochs))
        elif name == 'lce_m':
            scorers.append(cls(target_epochs=target_epochs))
        else:
            scorers.append(cls())
    return scorers
