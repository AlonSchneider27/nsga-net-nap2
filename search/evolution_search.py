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

parser = argparse.ArgumentParser("Multi-objetive Genetic Algorithm for NAS")
parser.add_argument('--save', type=str, default='GA-BiObj', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--search_space', type=str, default='micro', help='macro or micro search space')
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
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

pop_hist = []  # keep track of every evaluated architecture


# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, search_space='micro', n_var=20, n_obj=1, n_constr=0, lb=None, ub=None,
                 init_channels=24, layers=8, epochs=25, save_dir=None, predictor=None,
                 dataset='cifar10'):
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
            performance = train_search.main(genome=genome,
                                            search_space=self._search_space,
                                            init_channels=self._init_channels,
                                            layers=self._layers, cutout=False,
                                            epochs=self._epochs,
                                            save='arch_{}'.format(arch_id),
                                            expr_root=self._save_dir,
                                            predictor=self._predictor,
                                            dataset=self._dataset)

            # all objectives assume to be MINIMIZED !!!!!
            objs[i, 0] = 100 - performance['valid_acc']
            objs[i, 1] = performance['flops']

            pred_acc = performance.get('pred_acc')
            pred_str = '{:.4f}'.format(pred_acc) if pred_acc is not None else 'n/a'
            logging.info('arch %d: valid_acc=%.4f pred_acc=%s', arch_id, performance['valid_acc'], pred_str)

            self._n_evaluated += 1

        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
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
    from nap2.autoencoder import FeatureMapAutoEncoder
    from nap2.lstm_predictor import LSTMPredictor
    from nap2.bigru_predictor import BiGRUDualPredictor

    normalize = 'none'
    if paths['ae_weights_json']:
        with open(paths['ae_weights_json']) as f:
            normalize = json.load(f).get('normalize', 'none')

    with open(paths['lstm_json']) as f:
        predictor_type = json.load(f).get('predictor_type', 'lstm')

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

    predictor = None
    if args.use_nap2:
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
    else:
        raise NameError('Unknown search space type')

    problem = NAS(n_var=n_var, search_space=args.search_space,
                  n_obj=2, n_constr=0, lb=lb, ub=ub,
                  init_channels=args.init_channels, layers=args.layers,
                  epochs=args.epochs, save_dir=args.save,
                  predictor=predictor, dataset=args.dataset)

    # configure the nsga-net method
    method = engine.nsganet(pop_size=args.pop_size,
                            n_offsprings=args.n_offspring,
                            eliminate_duplicates=True)

    res = minimize(problem,
                   method,
                   callback=do_every_generations,
                   termination=('n_gen', args.n_gens))

    return


if __name__ == "__main__":
    main()