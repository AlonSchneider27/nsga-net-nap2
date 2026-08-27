import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import time
import logging
import argparse
from misc import utils

import numpy as np
from search import train_search
from search import micro_encoding
from search import macro_encoding
from search import nb201_encoding
from models.micro_genotypes import NB201_PRIMITIVES
from search import nsganet as engine

from pymop.problem import Problem
from pymoo.optimize import minimize


# ----------------------------------------------------------------
# nap2 predictor checkpoint paths — paste here OR override via CLI.
# CLI flags (--nap2_ae_weights_pt, etc.) take precedence over these.
# Required when --use_nap2 is set; ignored otherwise. The two AE
# JSONs are optional (defaults are used if empty); the four .pt
# paths and the predictor JSON are required.
# ----------------------------------------------------------------
NAP2_AE_WEIGHTS_PT     = ''
NAP2_AE_WEIGHTS_JSON   = ''   # AE hyperparams; carries 'layers_shapes' and (optional) 'normalize'
NAP2_AE_GRADIENTS_PT   = ''
NAP2_AE_GRADIENTS_JSON = ''   # same, for the gradients AE
NAP2_LSTM_PT           = ''
NAP2_LSTM_JSON         = ''   # predictor hyperparams; carries 'predictor_type' (lstm|bigru)
# LC-PFN pretrained checkpoint (see scripts/fetch_lcpfn_checkpoint.sh).
# Required when --fitness includes lc_pfn; --lcpfn_ckpt overrides.
LCPFN_CKPT             = ''

parser = argparse.ArgumentParser("Multi-objetive Genetic Algorithm for NAS")
parser.add_argument('--save', type=str, default='GA-BiObj', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--search_space', type=str, default='micro',
                    choices=['micro', 'macro', 'nb201'],
                    help='search space: micro (DARTS-style cells), macro (GeneticCNN-style), '
                         'or nb201 (NAS-Bench-201 5-op DAG cells)')
# arguments for micro search space
parser.add_argument('--n_blocks', type=int, default=5, help='number of blocks in a cell')
parser.add_argument('--n_ops', type=int, default=9, help='number of operations considered')
parser.add_argument('--n_cells', type=int, default=2, help='number of cells to search')
# arguments for macro search space
parser.add_argument('--n_nodes', type=int, default=4, help='number of nodes per phases')
# hyper-parameters for algorithm
parser.add_argument('--pop_size', type=int, default=40, help='population size of networks')
parser.add_argument('--n_gens', type=int, default=50, help='population size')
parser.add_argument('--n_offspring', type=int, default=40, help='number of offspring created per generation')
# arguments for back-propagation training during search
parser.add_argument('--init_channels', type=int, default=24, help='# of filters for first cell')
parser.add_argument('--layers', type=int, default=11, help='equivalent with N = 3')
parser.add_argument('--epochs', type=int, default=25, help='# of epochs to train during architecture search')
parser.add_argument('--output_dir', type=str, default='.', help='parent directory under which the experiment folder is created')
parser.add_argument('--use_nap2', action='store_true', help='collect nap2 predicted accuracy alongside training (log-only, does not affect GA objectives)')
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'ImageNet16-120'],
                    help='dataset for the search-phase proxy training')
# nap2 checkpoint paths — each defaults to '' so we can detect whether the
# user supplied them. They override the matching module-level NAP2_* constant.
parser.add_argument('--nap2_ae_weights_pt', type=str, default='',
                    help='path to the AE-weights .pt checkpoint (required with --use_nap2)')
parser.add_argument('--nap2_ae_weights_json', type=str, default='',
                    help='path to the AE-weights JSON hyperparams (optional; uses defaults if empty)')
parser.add_argument('--nap2_ae_gradients_pt', type=str, default='',
                    help='path to the AE-gradients .pt checkpoint (required with --use_nap2)')
parser.add_argument('--nap2_ae_gradients_json', type=str, default='',
                    help='path to the AE-gradients JSON hyperparams (optional; uses defaults if empty)')
parser.add_argument('--nap2_lstm_pt', type=str, default='',
                    help='path to the predictor (LSTM or BiGRU) .pt checkpoint (required with --use_nap2)')
parser.add_argument('--nap2_lstm_json', type=str, default='',
                    help='path to the predictor JSON hyperparams; predictor_type detected from this file (required with --use_nap2)')
parser.add_argument('--data', type=str, default='',
                    help='dataset root directory. Empty = the default in misc/dataset_configs.py '
                         '(cifar10/cifar100 auto-download there; ImageNet16-120 needs an existing '
                         'dir with either the NB201 pickle batches or the .npy layout '
                         'x_train/y_train/x_val/y_val).')
parser.add_argument('--nap2_steps', type=int, default=5,
                    help='number of snapshots nap2 collects per architecture; each snapshot costs snapshot_interval (default 100) mini-batches of partial training')
parser.add_argument('--nap2_max_steps', type=int, default=0,
                    help='zero-pad the nap2 embedding sequence to this length before prediction. '
                         'Default 0 = no padding: the deployed BiGRU was trained on length-5 '
                         'sequences (verified against michael\'s embedding cache '
                         'cifar10_via_log_cifar10.pkl, [15625 x 5 x 256]), so a plain '
                         '[steps, 256] sequence is the in-distribution input. Only set this '
                         'if a future predictor is trained on longer padded sequences.')
parser.add_argument('--fitness', type=str, default='',
                    help='comma-separated fitness methods to score and log per architecture: '
                         'any of nap2,synflow,grad_norm,snip,sotl,sotl_e,early_stop,lce_m,lc_pfn, '
                         'or "all" for the eight baselines (add nap2 with "all,nap2"). Scores are '
                         'shadow-logged for post-hoc Kendall-tau comparison; the GA objectives '
                         'stay 100-valid_acc and flops. Learning-curve methods share one '
                         'partial-training run at the --nap2_steps budget; zero-cost proxies '
                         '(synflow/grad_norm/snip) score at initialization, budget-free.')
parser.add_argument('--nap2_steps_list', type=str, default='',
                    help='comma-separated snapshot budgets, e.g. "1,3,5,10,15". ONE '
                         'partial-training run per architecture at max(list); every '
                         'fitness method AND nap2 are then scored at each budget from '
                         'prefixes of that run, logged with @budget-suffixed keys '
                         '(sotl@3=..., nap2@15=...) so summary.json reports KT per '
                         '(method, budget). Zero-cost proxies stay budget-free (plain '
                         'keys). Empty (default) = single budget from --nap2_steps '
                         'with plain keys, exactly as before.')
parser.add_argument('--lcpfn_ckpt', type=str, default='',
                    help='path to the LC-PFN pretrained checkpoint '
                         '(scripts/fetch_lcpfn_checkpoint.sh; required when --fitness '
                         'includes lc_pfn). Overrides the LCPFN_CKPT module constant.')
parser.add_argument('--lc_target_epochs', type=int, default=0,
                    help='epoch horizon lce_m/lc_pfn extrapolate the val-acc curve to. '
                         'Default 0 = use --epochs (the horizon the summary KT is computed '
                         'against); set 200 for the NB201 full-training horizon.')
parser.add_argument('--lc_cadence', type=str, default='snapshot',
                    choices=['snapshot', 'epoch'],
                    help='sampling cadence for the 5 learning-curve methods. '
                         '"snapshot" (default) = one observation per 100 '
                         'mini-batches from a shared partial-training run '
                         '(the paper-table axis). "epoch" = NATIVE cadence: '
                         'signals read from the real proxy training loop '
                         '(per-epoch loss sums + one val pass per epoch), '
                         'keys logged as name@e<K> for every epoch K; '
                         'requires --epochs > 0. nap2 (100-mb snapshots) and '
                         'zero-cost proxies (at-init) keep their own native '
                         'cadence either way.')
parser.add_argument('--fitness_objective', type=str, default='',
                    help='single fitness method whose score REPLACES the accuracy '
                         'objective: objs[0] = -score (higher=better; pymoo minimizes). '
                         'One of nap2,synflow,grad_norm,snip,sotl,sotl_e,early_stop,'
                         'lce_m,lc_pfn. Auto-added to the scored set; --fitness still '
                         'shadow-logs anything else listed. Score taken at --nap2_steps, '
                         'or max(--nap2_steps_list) when set. Failed/non-finite scores '
                         'get a large penalty (arch deselected). Default "" = objectives '
                         'unchanged (100-valid_acc, flops). Combine with --epochs 0 to '
                         'skip per-arch proxy training entirely.')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

pop_hist = []  # keep track of every evaluated architecture


# Guided mode: an arch whose objective score failed gets this value for
# objs[0] so NSGA-II deselects it (must exceed any plausible -score; sotl's
# -score is a loss sum that can reach the low thousands).
OBJECTIVE_PENALTY = 1e9


def objective_score(performance, method, steps_list=None):
    """Guided-objective score for `method` from one train_search.main() result.

    Returns a float, or None when the score is missing. 'nap2' reads
    performance['pred_acc'] (which already equals the max-budget nap2@k in
    budget-list mode); other methods read fitness_scores at
    '<name>@<max budget>' with a plain-key fallback (zero-cost proxies stay
    budget-free even in list mode).
    """
    if method == 'nap2':
        return performance.get('pred_acc')
    fs = performance.get('fitness_scores') or {}
    if steps_list:
        return fs.get('{}@{}'.format(method, steps_list[-1]), fs.get(method))
    return fs.get(method)


# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, search_space='micro', n_var=20, n_obj=1, n_constr=0, lb=None, ub=None,
                 init_channels=24, layers=8, epochs=25, save_dir=None, predictor=None,
                 dataset='cifar10', data='', nap2_steps=5, nap2_max_steps=0,
                 fitness_scorers=None, nap2_steps_list=None, fitness_objective='',
                 lc_cadence='snapshot'):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int)
        self.xl = lb
        self.xu = ub
        self._search_space = search_space
        self._init_channels = init_channels
        self._layers = layers
        self._epochs = epochs
        self._save_dir = save_dir
        self._predictor = predictor
        self._dataset = dataset
        self._data = data
        self._nap2_steps = nap2_steps
        self._nap2_max_steps = nap2_max_steps
        self._fitness_scorers = fitness_scorers
        self._nap2_steps_list = nap2_steps_list
        self._fitness_objective = fitness_objective
        self._lc_cadence = lc_cadence
        # Genome-keyed cache: pymoo dedups offspring within a generation but
        # re-samples across generations, and every re-evaluation costs a full
        # proxy training. Budget and method set are constant within a run, so
        # caching the whole performance dict is safe.
        self._perf_cache = {}
        self._n_evaluated = 0  # keep track of how many architectures are sampled

    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_obj), np.nan)

        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1
            print('\n')
            logging.info('Network id = {}'.format(arch_id))

            # call back-propagation training
            if self._search_space == 'micro':
                genome = micro_encoding.convert(x[i, :])
            elif self._search_space == 'macro':
                genome = macro_encoding.convert(x[i, :])
            elif self._search_space == 'nb201':
                genome = nb201_encoding.convert(x[i, :])
            cache_key = str(genome)
            cached = self._perf_cache.get(cache_key)
            if cached is not None:
                performance = cached
                # Re-log the scrapeable per-arch lines so this arch_id still
                # appears fully in summary.json.
                logging.info('arch %d: cache hit (genome already evaluated)', arch_id)
                logging.info("Architecture = %s", performance['genotype'])
                logging.info("param size = %fMB", performance['params'])
                logging.info('flops = %f', performance['flops'])
            else:
                performance = train_search.main(genome=genome,
                                                search_space=self._search_space,
                                                init_channels=self._init_channels,
                                                layers=self._layers, cutout=False,
                                                epochs=self._epochs,
                                                save='arch_{}'.format(arch_id),
                                                expr_root=self._save_dir,
                                                predictor=self._predictor,
                                                dataset=self._dataset,
                                                data=self._data,
                                                nap2_steps=self._nap2_steps,
                                                nap2_max_steps=self._nap2_max_steps,
                                                fitness_scorers=self._fitness_scorers,
                                                nap2_steps_list=self._nap2_steps_list,
                                                lc_cadence=self._lc_cadence)
                # Guided mode: don't cache a result whose guiding score
                # failed — caching it would turn a transient failure (OOM,
                # predictor exception) into a permanent penalty against that
                # genome for the whole search.
                if self._fitness_objective:
                    _s = objective_score(performance, self._fitness_objective,
                                         self._nap2_steps_list)
                    if _s is not None and np.isfinite(_s):
                        self._perf_cache[cache_key] = performance
                else:
                    self._perf_cache[cache_key] = performance

            # all objectives assume to be MINIMIZED !!!!!
            if self._fitness_objective:
                score = objective_score(performance, self._fitness_objective,
                                        self._nap2_steps_list)
                if score is None or not np.isfinite(score):
                    objs[i, 0] = OBJECTIVE_PENALTY
                    logging.warning('arch %d: %s score unavailable (%r); '
                                    'assigning penalty %g', arch_id,
                                    self._fitness_objective, score,
                                    OBJECTIVE_PENALTY)
                else:
                    objs[i, 0] = -score
            else:
                objs[i, 0] = 100 - performance['valid_acc']
            objs[i, 1] = performance['flops']

            pred_acc = performance.get('pred_acc')
            pred_str = '{:.4f}'.format(pred_acc) if pred_acc is not None else 'n/a'
            logging.info('arch %d: valid_acc=%.4f pred_acc=%s', arch_id, performance['valid_acc'], pred_str)

            fitness_scores = performance.get('fitness_scores')
            if fitness_scores:
                pairs = ' '.join('{}={:.6f}'.format(k, v)
                                 for k, v in fitness_scores.items() if v is not None)
                if pairs:
                    logging.info('arch %d fitness: %s', arch_id, pairs)

            self._n_evaluated += 1

        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def save_final_population(res, search_space, path, objective_method=''):
    """Write the final population (+ non-dominated-front membership) as JSON.

    `res` is a pymoo 0.3.0 Result: .pop is the final Population, .X the
    decision variables of the feasible non-dominated front (None when empty).
    The population was previously discarded; guided runs need it to pick
    which architectures to fully train afterwards. Returns the payload.
    """
    import json
    encoders = {'micro': micro_encoding, 'macro': macro_encoding,
                'nb201': nb201_encoding}
    enc = encoders[search_space]

    front_keys = set()
    if getattr(res, 'X', None) is not None:
        for row in np.atleast_2d(res.X):
            front_keys.add(str(enc.convert(row)))

    population = []
    pop_x = res.pop.get('X')
    pop_f = res.pop.get('F')
    for row, f in zip(pop_x, pop_f):
        genome = enc.convert(row)
        genotype = enc.decode(genome)
        entry = {
            'X': [int(v) for v in np.asarray(row).ravel()],
            'F': [float(v) for v in np.asarray(f).ravel()],
            'genotype': repr(genotype),
            'on_pareto_front': str(genome) in front_keys,
        }
        if search_space == 'nb201':
            entry['arch_str'] = genotype.arch_str
        population.append(entry)

    payload = {
        'search_space': search_space,
        'objective_method': objective_method,
        'objectives': ['-{}'.format(objective_method) if objective_method
                       else '100-valid_acc', 'flops'],
        'population': population,
    }
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    return payload


def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")

    # report generation info to files
    logging.info("generation = {}".format(gen))
    logging.info("population error: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                  np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    logging.info("population complexity: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 1]), np.mean(pop_obj[:, 1]),
                                                  np.median(pop_obj[:, 1]), np.max(pop_obj[:, 1])))


# ---------------------------------------------------------------------------------------------------------
# nap2 predictor loading: resolve per-file paths (CLI > module constant) and
# build a NAP2Predictor by direct construction. Bypasses NAP2Predictor.load(),
# which insists on a fixed sub-tree layout — too rigid for our use case where
# checkpoint filenames vary.
# ---------------------------------------------------------------------------------------------------------
def _resolve_nap2_paths(args):
    """Return resolved nap2 checkpoint paths.

    CLI flag overrides take precedence over the module-level NAP2_* constants.
    The four .pt paths and the predictor JSON are required; the two AE JSONs
    are optional (loader falls back to defaults when empty).
    """
    paths = {
        'ae_weights_pt':     args.nap2_ae_weights_pt     or NAP2_AE_WEIGHTS_PT,
        'ae_weights_json':   args.nap2_ae_weights_json   or NAP2_AE_WEIGHTS_JSON,
        'ae_gradients_pt':   args.nap2_ae_gradients_pt   or NAP2_AE_GRADIENTS_PT,
        'ae_gradients_json': args.nap2_ae_gradients_json or NAP2_AE_GRADIENTS_JSON,
        'lstm_pt':           args.nap2_lstm_pt           or NAP2_LSTM_PT,
        'lstm_json':         args.nap2_lstm_json         or NAP2_LSTM_JSON,
    }
    required = ('ae_weights_pt', 'ae_gradients_pt', 'lstm_pt', 'lstm_json')
    missing = [k for k in required if not paths[k]]
    if missing:
        flags = ', '.join('--nap2_' + k for k in missing)
        raise ValueError(
            "--use_nap2 requires the following paths (set the matching "
            "NAP2_* constants at the top of search/evolution_search.py "
            "or pass {}).".format(flags)
        )
    return paths


def _load_nap2_predictor(paths):
    """Build a NAP2Predictor from explicit per-file paths.

    Mirrors NAP2Predictor.load()'s behavior (auto-detects predictor type
    and normalization) but without enforcing a directory layout.
    """
    import json
    from nap2 import NAP2Predictor
    from nap2.predictor import resolve_normalize
    from nap2.autoencoder import FeatureMapAutoEncoder
    from nap2.lstm_predictor import LSTMPredictor
    from nap2.bigru_predictor import BiGRUDualPredictor

    ae_params = None
    if paths['ae_weights_json']:
        with open(paths['ae_weights_json']) as f:
            ae_params = json.load(f)

    with open(paths['lstm_json']) as f:
        pred_params = json.load(f)
    predictor_type = pred_params.get('predictor_type', 'lstm')

    # The AEs were trained on log-transformed feature maps; inference must
    # apply the same transform. The value lives under different keys across
    # checkpoint sets, so resolve it from all of them and log the origin --
    # a silent fallback to 'none' disables the transform and collapses every
    # prediction to the predictor's prior.
    normalize, normalize_src = resolve_normalize(ae_params, pred_params)
    logging.info('nap2 feature-map normalize=%s (from %s)', normalize, normalize_src)
    if normalize == 'none':
        logging.warning(
            'nap2: normalize resolved to "none" -- no log transform will be '
            'applied. If these AEs were trained on log-transformed maps '
            '(controlled_aes/log_transform/...), predictions will be garbage.'
        )

    ae_w = FeatureMapAutoEncoder.load(
        model_path=paths['ae_weights_pt'],
        params_path=paths['ae_weights_json'] or None,
    )
    ae_g = FeatureMapAutoEncoder.load(
        model_path=paths['ae_gradients_pt'],
        params_path=paths['ae_gradients_json'] or None,
    )
    PredictorCls = BiGRUDualPredictor if predictor_type == 'bigru' else LSTMPredictor
    pred = PredictorCls.load(model_path=paths['lstm_pt'], params_path=paths['lstm_json'])

    return NAP2Predictor(ae_weights=ae_w, ae_gradients=ae_g, lstm=pred, normalize=normalize)


def main():
    args.save = os.path.join(args.output_dir, 'search-{}-{}-{}-{}'.format(args.save, args.search_space, args.dataset, time.strftime("%Y%m%d-%H%M%S")))
    utils.create_exp_dir(args.save)
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    np.random.seed(args.seed)
    logging.info("args = %s", args)

    # Parse --fitness: expand 'all' to the five baselines; 'nap2' routes to
    # the existing predictor path (identical to --use_nap2).
    import fitness as fitness_pkg
    tokens = [t for t in (s.strip() for s in args.fitness.split(',')) if t]
    expanded = []
    for t in tokens:
        expanded.extend(fitness_pkg.ALL_BASELINES if t == 'all' else [t])
    LC_METHODS = {'sotl', 'sotl_e', 'early_stop', 'lce_m', 'lc_pfn'}
    if args.lc_cadence == 'epoch':
        if args.epochs == 0 and any(t in LC_METHODS for t in expanded):
            raise ValueError('--lc_cadence epoch reads signals from the real '
                             'training loop and needs --epochs > 0')
        if args.fitness_objective in LC_METHODS:
            raise ValueError('--fitness_objective with an LC method requires '
                             'the snapshot cadence (epoch-native scores are '
                             'shadow-logged only)')
        logging.info('lc_cadence = epoch (native: LC signals from the real '
                     'training loop, keys name@e<K>)')

    if args.fitness_objective:
        valid = set(fitness_pkg.ALL_BASELINES) | {'nap2'}
        if args.fitness_objective not in valid:
            raise ValueError('--fitness_objective must be one of {}, got {!r}'
                             .format(sorted(valid), args.fitness_objective))
        # The guiding method must be scored even when --fitness is empty.
        if args.fitness_objective not in expanded:
            expanded.append(args.fitness_objective)

    wants_nap2 = args.use_nap2 or 'nap2' in expanded
    baseline_names = [t for t in expanded if t != 'nap2']

    # lce_m/lc_pfn extrapolation horizon: --lc_target_epochs, else --epochs,
    # else 20 (the project GT horizon) — --epochs 0 guided runs must not
    # collapse the horizon to a single snapshot ahead.
    lc_horizon = args.lc_target_epochs or args.epochs or 20
    if args.epochs == 0 and not args.lc_target_epochs:
        logging.warning('--epochs 0: lce_m/lc_pfn extrapolation horizon '
                        'defaults to %d epochs; set --lc_target_epochs to '
                        'override', lc_horizon)

    fitness_scorers = None
    if baseline_names:
        fitness_scorers = fitness_pkg.build_scorers(
            ','.join(baseline_names),
            lcpfn_ckpt=args.lcpfn_ckpt or LCPFN_CKPT,
            target_epochs=lc_horizon)
        logging.info('fitness baselines enabled: %s (lc target horizon: %d epochs)',
                     [s.name for s in fitness_scorers], lc_horizon)

    # Parse --nap2_steps_list into sorted unique positive ints (or None).
    nap2_steps_list = None
    if args.nap2_steps_list:
        nap2_steps_list = sorted({int(s) for s in args.nap2_steps_list.split(',')
                                  if s.strip()})
        if not nap2_steps_list or any(k < 1 for k in nap2_steps_list):
            raise ValueError(f'--nap2_steps_list must be positive ints, '
                             f'got {args.nap2_steps_list!r}')
        logging.info('budget list enabled: %s snapshots (partial train at %d, '
                     'all methods scored per budget from prefixes)',
                     nap2_steps_list, nap2_steps_list[-1])

    if args.fitness_objective:
        logging.info('objective_method = %s (objs[0] = -score at budget %s; '
                     'flops objective unchanged)', args.fitness_objective,
                     nap2_steps_list[-1] if nap2_steps_list else args.nap2_steps)

    predictor = None
    if wants_nap2:
        paths = _resolve_nap2_paths(args)
        predictor = _load_nap2_predictor(paths)
        logging.info("nap2 predictor loaded (lstm=%s, ae_w=%s, ae_g=%s)",
                     paths['lstm_pt'], paths['ae_weights_pt'], paths['ae_gradients_pt'])

    # setup NAS search problem
    if args.search_space == 'micro':  # NASNet search space
        n_var = int(4 * args.n_blocks * 2)
        lb = np.zeros(n_var)
        ub = np.ones(n_var)
        h = 1
        for b in range(0, n_var//2, 4):
            ub[b] = args.n_ops - 1
            ub[b + 1] = h
            ub[b + 2] = args.n_ops - 1
            ub[b + 3] = h
            h += 1
        ub[n_var//2:] = ub[:n_var//2]
    elif args.search_space == 'macro':  # modified GeneticCNN search space
        n_var = int(((args.n_nodes-1)*args.n_nodes/2 + 1)*3)
        lb = np.zeros(n_var)
        ub = np.ones(n_var)
    elif args.search_space == 'nb201':   # NAS-Bench-201 5-op DAG cells
        # 6 edges, each picking one of len(NB201_PRIMITIVES) ops.
        # Total design space: len(NB201_PRIMITIVES)**6 = 5**6 = 15_625,
        # which equals the published NB201 catalog.
        #
        # dtype=float is required: pymoo's polynomial_mutation does
        # ``xu += 0.5`` in-place to handle integer-coded variables.
        # A bare ``np.full(n, 4)`` produces an int64 array, and numpy
        # refuses the in-place float -> int cast (same_kind rule),
        # crashing pymoo on the first mutation step. micro/macro
        # sidestep this by starting from ``np.ones`` (which is float64).
        n_var = nb201_encoding.N_EDGES
        lb = np.zeros(n_var)
        ub = np.full(n_var, len(NB201_PRIMITIVES) - 1, dtype=float)
    else:
        raise NameError('Unknown search space type')

    problem = NAS(n_var=n_var, search_space=args.search_space,
                  n_obj=2, n_constr=0, lb=lb, ub=ub,
                  init_channels=args.init_channels, layers=args.layers,
                  epochs=args.epochs, save_dir=args.save,
                  predictor=predictor, dataset=args.dataset,
                  data=args.data,
                  nap2_steps=args.nap2_steps,
                  nap2_max_steps=args.nap2_max_steps,
                  fitness_scorers=fitness_scorers,
                  nap2_steps_list=nap2_steps_list,
                  fitness_objective=args.fitness_objective,
                  lc_cadence=args.lc_cadence)

    # configure the nsga-net method
    method = engine.nsganet(pop_size=args.pop_size,
                            n_offsprings=args.n_offspring,
                            eliminate_duplicates=True)

    res = minimize(problem,
                   method,
                   callback=do_every_generations,
                   termination=('n_gen', args.n_gens))

    # Persist the final population (previously discarded) so guided runs can
    # pick which architectures to fully train afterwards.
    try:
        payload = save_final_population(
            res, args.search_space, os.path.join(args.save, 'final_pop.json'),
            objective_method=args.fitness_objective)
        n_front = sum(1 for e in payload['population'] if e['on_pareto_front'])
        logging.info('final population written to final_pop.json '
                     '(%d individuals, %d on front)',
                     len(payload['population']), n_front)
    except Exception:
        logging.exception('final population save failed')

    # Write a per-architecture summary.json next to log.txt at run end.
    # Wrapped in try/except so a scrape failure can never invalidate the
    # search results that just finished.
    try:
        from misc.log_summary import write_summary
        log_path = os.path.join(args.save, 'log.txt')
        summary_path = os.path.join(args.save, 'summary.json')
        data = write_summary(log_path, summary_path)
        logging.info('summary written to %s (%d architectures)',
                     summary_path, len(data))
    except Exception:
        logging.exception('summary generation failed')

    return


if __name__ == "__main__":
    main()