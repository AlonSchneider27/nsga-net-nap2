# Vendored: lcpfn v0.1.3

Source: https://github.com/automl/lcpfn @ ba892f6f451027f69c50edf00c765ded98c75d30
(MIT license, see LICENSE). Paper: Adriaensen et al., "Efficient Bayesian Learning
Curve Extrapolation using Prior-Data Fitted Networks", arXiv:2310.20447.

Why vendored instead of `pip install lcpfn`:
- its pyproject pins `torch<=1.11.0`; this repo runs torch 2.x. The pin is
  packaging conservatism, not a runtime requirement.
- the pretrained checkpoint unpickles a whole nn.Module whose pickle references
  bare top-level module names (`transformer`, `bar_distribution`, ...); the
  package's own `sys.path.insert(0, ...)` in `__init__.py` makes those resolve,
  so the package must be importable as `lcpfn` — repo root works.

Deviations from upstream (marked `# VENDORED PATCH` in the code):
1. `model.py`: `torch.load(..., map_location="cpu", weights_only=False)` —
   torch>=2.6 defaults to `weights_only=True`, which refuses this checkpoint.

Checkpoint (not committed; ~100 MB): run `scripts/fetch_lcpfn_checkpoint.sh`,
which places it under `trained_models/lcpfn/` (gitignored). Pass that path as
`LCPFN(model_name=<path>)` — used by `fitness/lcpfn_scorer.py`.
