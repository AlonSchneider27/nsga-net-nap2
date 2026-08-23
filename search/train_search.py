import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
# import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models.macro_models import EvoNetwork
from models.micro_models import NetworkCIFAR as Network

import time
from misc import utils
from misc.dataset_configs import get_config, get_loader_class, build_search_transforms
from search import micro_encoding
from search import macro_encoding
from misc.flops_counter import add_flops_counting_methods


if torch.cuda.is_available():
    device = 'cuda'
elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


class _LogitsOnly(nn.Module):
    """Adapter that exposes only the classification logits.

    NetworkCIFAR.forward returns ``(logits, aux_logits)`` with logits first,
    but nap2's _partial_train assumes the NAS-Bench-201 convention
    ``(features, logits)`` and takes ``outputs[-1]`` (which is the unused
    aux head, often None). Wrapping the model with this adapter normalizes
    the output to a single logits tensor before nap2 sees it.
    """

    def __init__(self, m):
        super().__init__()
        self.inner = m

    def forward(self, x):
        out = self.inner(x)
        return out[0] if isinstance(out, tuple) else out


class _LogitsFirstFromNB201(nn.Module):
    """Adapter that flips NB201's (features, logits) to (logits, None).

    NSGA-Net's train()/infer() unpack as ``outputs, outputs_aux = net(inputs)``
    and pass ``outputs`` into the loss. NetworkCIFAR returns (logits, aux),
    but NB201's TinyNetwork returns (features, logits) — the opposite
    convention. Without this adapter, train() would compute the loss on
    features, not logits.
    """

    def __init__(self, m):
        super().__init__()
        self.inner = m

    def forward(self, x):
        out = self.inner(x)
        # NB201 always returns a 2-tuple (features, logits); take logits.
        features, logits = out
        return logits, None


def main(genome, epochs, search_space='micro',
         save='Design_1', expr_root='search', seed=0, gpu=0, init_channels=24,
         layers=11, auxiliary=False, cutout=False, drop_path_prob=0.0, predictor=None,
         dataset='cifar10', data='', nap2_steps=5, nap2_max_steps=0,
         fitness_scorers=None, nap2_steps_list=None):

    # ---- train logger ----------------- #
    save_pth = os.path.join(expr_root, '{}'.format(save))
    utils.create_exp_dir(save_pth)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    # fh = logging.FileHandler(os.path.join(save_pth, 'log.txt'))
    # fh.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(fh)

    # ---- dataset config -------------- #
    cfg = get_config(dataset)
    num_classes = cfg['num_classes']
    data_root = data if data else cfg['data_dir']
    image_size = cfg['image_size']

    # ---- parameter values setting ----- #
    learning_rate = 0.025
    momentum = 0.9
    weight_decay = 3e-4
    batch_size = 128
    nesterov = False
    t_max = epochs
    cutout_length = 16
    auxiliary_weight = 0.4
    grad_clip = 5
    report_freq = 50

    # NB201 ground-truth recipe (Dong & Yang, ICLR'20): SGD w/ Nesterov
    # momentum, lr 0.1, weight decay 5e-4, batch 256, grad clip 5, and a
    # cosine schedule over NB201's full 200-epoch length (T_max=200). We run
    # far fewer epochs, so the LR follows only the opening (near-flat) slice
    # of NB201's real schedule rather than fully annealing. The nap2 predictor
    # scores on the same 256-batch training queue, matching the batch size its
    # snapshots were generated at. cutout / auxiliary head / drop-path stay off
    # (already disabled here), matching NB201's plain augmentation.
    if search_space == 'nb201':
        learning_rate = 0.1
        weight_decay = 5e-4
        batch_size = 256
        nesterov = True
        t_max = 200

    train_params = {
        'auxiliary': auxiliary,
        'auxiliary_weight': auxiliary_weight,
        'grad_clip': grad_clip,
        'report_freq': report_freq,
    }

    if search_space == 'micro':
        genotype = micro_encoding.decode(genome)
        model = Network(init_channels, num_classes, layers, auxiliary, genotype)
    elif search_space == 'macro':
        genotype = macro_encoding.decode(genome)
        channels = [(3, init_channels),
                    (init_channels, 2*init_channels),
                    (2*init_channels, 4*init_channels)]
        model = EvoNetwork(genotype, channels, num_classes, image_size, decoder='residual')
    elif search_space == 'nb201':
        # NAS-Bench-201 5-op DAG cells. Reuses nap2's TinyNetwork builder
        # (single Conv2d stem, works on both 32x32 and 16x16 inputs).
        from search import nb201_encoding
        from nap2.search_spaces.nb201_ops import build_nb201_model
        genotype = nb201_encoding.decode(genome)
        # NB201 has 3 stages; --layers maps to the total number of cells,
        # so cells-per-stage = layers // 3. Floor of 1 keeps tiny smoke
        # runs (e.g. --layers 2) from collapsing.
        n_cells_per_stage = max(layers // 3, 1)
        if predictor is not None and n_cells_per_stage != 5:
            # nap2's snapshots were collected on standard NB201 nets (N=5,
            # build_nb201_model's default). Scoring a different depth feeds
            # the predictor weight/gradient statistics from a network it was
            # never trained on.
            logging.warning(
                'nap2: --layers %d gives N=%d cells/stage, but the predictor '
                'was trained on N=5 (NB201 standard). pred_acc will be '
                'unreliable; use --layers 15.',
                layers, n_cells_per_stage,
            )
        model = build_nb201_model(
            genotype.arch_str,
            num_classes=num_classes,
            C=init_channels,
            N=n_cells_per_stage,
        )
        # NB201 returns (features, logits); NSGA-Net's train/infer want
        # (logits, aux). Flip via adapter so the search loop is uniform.
        model = _LogitsFirstFromNB201(model)
    else:
        raise NameError('Unknown search space type')

    # logging.info("Genome = %s", genome)
    logging.info("Architecture = %s", genotype)

    if device == 'cuda':
        torch.cuda.set_device(gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    n_params = (np.sum(np.prod(v.size()) for v in filter(lambda p: p.requires_grad, model.parameters())) / 1e6)
    model = model.to(device)

    logging.info("param size = %fMB", n_params)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )

    train_transform = build_search_transforms(cfg, train=True)
    if cutout:
        # Insert Cutout right before Normalize (or last for unnormalized datasets).
        insert_at = -1 if cfg['mean'] is not None else len(train_transform.transforms)
        train_transform.transforms.insert(insert_at, utils.Cutout(cutout_length))

    valid_transform = build_search_transforms(cfg, train=False)

    DatasetCls = get_loader_class(dataset)
    train_data = DatasetCls(root=data_root, train=True, download=True, transform=train_transform)
    valid_data = DatasetCls(root=data_root, train=False, download=True, transform=valid_transform)

    # num_train = len(train_data)
    # indices = list(range(num_train))
    # split = int(np.floor(train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=4)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(t_max))

    pred_acc = None
    nap2_budget_preds = None
    if predictor is not None:
        try:
            import copy
            # nap2's pipeline is CPU-resident (AE is float64 on CPU, snapshots
            # call .cpu().numpy()). Keep partial training on CPU too so dtypes
            # and devices stay consistent across the whole score path.
            # Deepcopy so nap2's partial training (SGD steps, BN updates) doesn't
            # leak into the model that's about to be trained. On MPS, fall back
            # to CPU because nap2's float64 AE + MPS produces NNPack dtype
            # mismatches; on CUDA/CPU, score on the same device as training.
            score_model = copy.deepcopy(model)
            if device == 'mps':
                score_model = score_model.cpu()
            # NetworkCIFAR.forward reads self.droprate; the training loop sets
            # it per epoch but we score before training starts.
            score_model.droprate = 0.0
            score_model = _LogitsOnly(score_model)
            if nap2_steps_list:
                # Budget-list mode: ONE snapshot collection at the largest
                # budget, then predict every prefix (per-budget padding
                # semantics identical to a single run at that budget).
                emb = predictor.get_embeddings(score_model, train_queue,
                                               steps=max(nap2_steps_list))
                nap2_budget_preds = {}
                for k in nap2_steps_list:
                    seq = emb[:k]
                    if nap2_max_steps and nap2_max_steps > seq.shape[0]:
                        pad = torch.zeros(nap2_max_steps - seq.shape[0],
                                          seq.shape[1], dtype=seq.dtype)
                        seq = torch.cat([seq, pad], dim=0)
                    nap2_budget_preds[k] = float(predictor._lstm.predict(seq))
                pred_acc = nap2_budget_preds[max(nap2_steps_list)]
                logging.info('nap2 pred_acc = %.4f (steps=%d, pad_to=%s; budgets %s)',
                             pred_acc, max(nap2_steps_list), nap2_max_steps,
                             ' '.join(f'@{k}={v:.4f}'
                                      for k, v in sorted(nap2_budget_preds.items())))
            else:
                pred_acc = float(predictor.score(score_model, train_queue, steps=nap2_steps,
                                                 max_steps=nap2_max_steps))
                logging.info('nap2 pred_acc = %.4f (steps=%d, pad_to=%s)',
                             pred_acc, nap2_steps, nap2_max_steps)
            del score_model
        except Exception:
            logging.exception('nap2 prediction failed')

    fitness_scores = None
    if fitness_scorers or nap2_budget_preds:
        import copy
        fitness_scores = {}
        # Budget-list mode: nap2's per-budget predictions join the fitness
        # dict as nap2@k so the summary computes KT per (method, budget).
        if nap2_budget_preds:
            for k, v in sorted(nap2_budget_preds.items()):
                fitness_scores[f'nap2@{k}'] = v
        init_scorers = [s for s in (fitness_scorers or [])
                        if getattr(s, 'needs_init_model', False)]
        trace_scorers = [s for s in (fitness_scorers or [])
                         if not getattr(s, 'needs_init_model', False)]

        # Zero-cost proxies (SynFlow, GradNorm, SNIP): score the untrained
        # network from one minibatch, before any partial training. Each
        # scorer deepcopies the model internally, so the copy here stays
        # pristine across scorers.
        if init_scorers:
            # Creating a DataLoader iterator draws a base seed from torch's
            # CPU RNG; save/restore the state so enabling zero-cost methods
            # cannot shift minibatch order (and thus valid_acc) downstream.
            rng_state = torch.get_rng_state()
            try:
                zc_model = copy.deepcopy(model)
                zc_model.droprate = 0.0
                zc_model = _LogitsOnly(zc_model)
                zc_inputs, zc_targets = next(iter(train_queue))
                zc_inputs = zc_inputs.to(device)
                zc_targets = zc_targets.to(device)
                for scorer in init_scorers:
                    try:
                        t0 = time.time()
                        value = float(scorer.score_init(zc_model, zc_inputs,
                                                        zc_targets))
                        fitness_scores[scorer.name] = value
                        logging.info('fitness[%s] = %.6f (at-init t_score=%.2fs)',
                                     scorer.name, value, time.time() - t0)
                    except Exception:
                        fitness_scores[scorer.name] = None
                        logging.exception('fitness[%s] scoring failed', scorer.name)
                del zc_model
            except Exception:
                for scorer in init_scorers:
                    fitness_scores.setdefault(scorer.name, None)
                logging.exception('zero-cost scoring setup failed')
            finally:
                torch.set_rng_state(rng_state)

        # Learning-curve baselines (SoTL, SoTL-E, Early-Stop, LCE-m, LC-PFN):
        # one shared partial-training run at the same budget nap2 scores at,
        # on a throwaway copy so the real training below starts fresh.
        if trace_scorers:
            try:
                from fitness.trace import prefix_trace, run_partial_train
                from nap2.predictor import DEFAULT_TRAINING_CONFIG
                interval = DEFAULT_TRAINING_CONFIG['snapshot_interval']
                multi = bool(nap2_steps_list)
                budgets = sorted(set(nap2_steps_list)) if multi else [nap2_steps]
                budget = budgets[-1] * interval
                score_model = copy.deepcopy(model)
                score_model.droprate = 0.0
                score_model = _LogitsOnly(score_model)
                # Sub-max budgets read early_stop from the val curve, so a
                # multi-budget run needs the curve whenever any scorer wants
                # a final val measurement.
                need_final = any(s.needs_final_val for s in trace_scorers)
                need_curve = any(s.needs_val_curve for s in trace_scorers) \
                    or (len(budgets) > 1 and need_final)
                trace = run_partial_train(
                    score_model, train_queue, valid_queue, budget,
                    need_val_curve=need_curve, need_final_val=need_final)
                for scorer in trace_scorers:
                    values = {}
                    t0 = time.time()
                    for k in budgets:
                        key = f'{scorer.name}@{k}' if multi else scorer.name
                        try:
                            sub = prefix_trace(trace, k) if multi else trace
                            value = float(scorer.score(sub))
                            fitness_scores[key] = value
                            values[k] = value
                        except Exception:
                            fitness_scores[key] = None
                            logging.exception('fitness[%s] scoring failed at '
                                              'budget %d', scorer.name, k)
                    t_score = time.time() - t0
                    logging.info(
                        'fitness[%s] = %s (t_train=%.1fs t_val=%.1fs t_score=%.2fs)',
                        scorer.name,
                        ' '.join(f'@{k*interval}mb={v:.6f}'
                                 for k, v in sorted(values.items())),
                        trace.times['train'], trace.times['val'], t_score)
                del score_model
            except Exception:
                logging.exception('fitness baseline partial training failed')

    for epoch in range(epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.droprate = drop_path_prob * epoch / epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer, train_params)
        logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    # calculate for flops
    model = add_flops_counting_methods(model)
    model.eval()
    model.start_flops_count()
    # Use the dataset's true image size so flops are correct on 16x16
    # ImageNet16-120 (was previously hardcoded to 32x32).
    random_data = torch.randn(1, 3, *image_size)
    model(torch.autograd.Variable(random_data).to(device))
    n_flops = np.round(model.compute_average_flops_cost() / 1e6, 4)
    logging.info('flops = %f', n_flops)

    # save to file
    # os.remove(os.path.join(save_pth, 'log.txt'))
    with open(os.path.join(save_pth, 'log.txt'), "w") as file:
        file.write("Genome = {}\n".format(genome))
        file.write("Architecture = {}\n".format(genotype))
        file.write("param size = {}MB\n".format(n_params))
        file.write("flops = {}MB\n".format(n_flops))
        file.write("valid_acc = {}\n".format(valid_acc))

    # logging.info("Architecture = %s", genotype))

    return {
        'valid_acc': valid_acc,
        'params': n_params,
        'flops': n_flops,
        'pred_acc': pred_acc,
        'fitness_scores': fitness_scores,
        # Repr of the decoded genotype so the evaluation cache can re-log
        # the scrapeable 'Architecture = ...' line on cache hits.
        'genotype': str(genotype),
    }


# def train(train_queue, model, criterion, optimizer, params):
#     objs = utils.AvgrageMeter()
#     top1 = utils.AvgrageMeter()
#     top5 = utils.AvgrageMeter()
#     model.train()
#
#     for step, (input, target) in enumerate(train_queue):
#         input = Variable(input).cuda()
#         target = Variable(target).cuda(async=True)
#
#         optimizer.zero_grad()
#         if params['auxiliary']:
#             logits, logits_aux = model(input)
#         else:
#             logits, _ = model(input)
#
#         loss = criterion(logits, target)
#         if params['auxiliary']:
#             loss_aux = criterion(logits_aux, target)
#             loss += params['auxiliary_weight'] * loss_aux
#         loss.backward()
#         nn.utils.clip_grad_norm(model.parameters(), params['grad_clip'])
#         optimizer.step()
#
#         prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
#         n = input.size(0)
#         objs.update(loss.data[0], n)
#         top1.update(prec1.data[0], n)
#         top5.update(prec5.data[0], n)
#
#         # if step % params['report_freq'] == 0:
#         #     logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
#
#     return top1.avg, objs.avg

# Training
def train(train_queue, net, criterion, optimizer, params):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for step, (inputs, targets) in enumerate(train_queue):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, outputs_aux = net(inputs)
        loss = criterion(outputs, targets)

        if params['auxiliary']:
            loss_aux = criterion(outputs_aux, targets)
            loss += params['auxiliary_weight'] * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), params['grad_clip'])
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    #     if step % args.report_freq == 0:
    #         logging.info('train %03d %e %f', step, train_loss/total, 100.*correct/total)
    #
    # logging.info('train acc %f', 100. * correct / total)

    return 100.*correct/total, train_loss/total


# def infer(valid_queue, model, criterion):
#     objs = utils.AvgrageMeter()
#     top1 = utils.AvgrageMeter()
#     top5 = utils.AvgrageMeter()
#     model.eval()
#
#     for step, (input, target) in enumerate(valid_queue):
#         input = Variable(input, volatile=True).cuda()
#         target = Variable(target, volatile=True).cuda(async=True)
#
#         logits, _ = model(input)
#
#         loss = criterion(logits, target)
#
#         prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
#         n = input.size(0)
#         objs.update(loss.data[0], n)
#         top1.update(prec1.data[0], n)
#         top5.update(prec5.data[0], n)
#
#         # if step % params['report_freq'] == 0:
#         #     logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
#
#     return top1.avg, objs.avg


def infer(valid_queue, net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # if step % args.report_freq == 0:
            #     logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)

    acc = 100.*correct/total
    # logging.info('valid acc %f', 100. * correct / total)

    return acc, test_loss/total


if __name__ == "__main__":
    DARTS_V2 = [[[[3, 0], [3, 1]], [[3, 0], [3, 1]], [[3, 1], [2, 0]], [[2, 0], [5, 2]]],
               [[[0, 0], [0, 1]], [[2, 2], [0, 1]], [[0, 0], [2, 2]], [[2, 2], [0, 1]]]]
    start = time.time()
    print(main(genome=DARTS_V2, epochs=20, save='DARTS_V2_16', seed=1, init_channels=16,
               auxiliary=False, cutout=False, drop_path_prob=0.0))
    print('Time elapsed = {} mins'.format((time.time() - start)/60))
    # start = time.time()
    # print(main(genome=DARTS_V2, epochs=20, save='DARTS_V2_32', seed=1, init_channels=32))
    # print('Time elapsed = {} mins'.format((time.time() - start) / 60))

