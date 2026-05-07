# NSGA-Net
Code accompanying the paper. All codes assume running from root directory. Please update the sys path at the beginning of the codes before running.
> [NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm](https://arxiv.org/abs/1810.03522)
>
> Zhichao Lu, Ian Whalen, Vishnu Boddeti, Yashesh Dhebar, Kalyanmoy Deb, Erik Goodman and Wolfgang Banzhaf
>
> *arXiv:1810.03522*

![overview](https://github.com/ianwhale/nsga-net/blob/beta/img/overview_redraw.png  "Overview of NSGA-Net")

## Requirements
``` 
Python >= 3.6.8, PyTorch >= 1.0.1.post2, torchvision >= 0.2.2, pymoo == 0.3.0
```

## Datasets

The search and validation phases support three datasets via the `--dataset` flag:

| Dataset | Classes | Image | Default data dir | Auto-download |
|---|---|---|---|---|
| `cifar10` (default) | 10 | 32×32 | `data/` | yes (cs.toronto.edu) |
| `cifar100` | 100 | 32×32 | `data/` | yes (cs.toronto.edu) |
| `ImageNet16-120` | 120 | 16×16 | `data/ImageNet16/` | yes if `IMAGENET16_URL` is set |

Expected data layout under the project root:

```
data/cifar-10-batches-py/                     (auto-downloaded)
data/cifar-100-python/                        (auto-downloaded)
data/ImageNet16/                  (canonical NB201 layout)
  ├── train_data_batch_1
  ├── train_data_batch_2
  ├── ...
  ├── train_data_batch_10
  └── val_data
```

ImageNet16-120 has no canonical public URL. Override the download mirror with the `IMAGENET16_URL` env var, or paste a working URL into `DEFAULT_URL` at the top of [search/imagenet16_search.py](search/imagenet16_search.py). If both auto-download and cache miss, the loader raises `FileNotFoundError` naming the exact paths and the URL it tried.

To hardcode an absolute data path (e.g. on a Slurm cluster), edit either `DEFAULT_ROOT` at the top of [search/imagenet16_search.py](search/imagenet16_search.py) or the `data_dir` field for `ImageNet16-120` in [misc/dataset_configs.py](misc/dataset_configs.py).

### Search-phase flags

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--dataset` | `cifar10` \| `cifar100` \| `ImageNet16-120` | `cifar10` | Selects classes, normalization, image size, and loader. |
| `--use_nap2` | (store_true) | off | Collect nap2 predicted accuracy alongside training (logged, doesn't change GA objectives). |
| `--search_space` | `micro` \| `macro` | `micro` | Architecture-search grammar. |
| `--init_channels` | int | `24` | Stem channel width. |
| `--layers` | int | `11` | Number of cells. |
| `--epochs` | int | `25` | Proxy training length per architecture. |
| `--pop_size` | int | `40` | NSGA-II population. |
| `--n_offspring` | int | `40` | Offspring per generation. |
| `--n_gens` | int | `50` | Generations. |
| `--output_dir` | path | `.` | Parent dir under which run folders are created. |

### Examples

CIFAR-10 with nap2 (regression check):

```bash
python search/evolution_search.py \
  --search_space micro --init_channels 16 --layers 8 \
  --epochs 20 --pop_size 40 --n_offspring 20 --n_gens 30 \
  --output_dir experiments/cifar10 \
  --use_nap2
```

CIFAR-100 with nap2 (matches the dataset our nap2 checkpoint was trained on):

```bash
python search/evolution_search.py \
  --dataset cifar100 \
  --search_space micro --init_channels 16 --layers 8 \
  --epochs 20 --pop_size 40 --n_offspring 20 --n_gens 30 \
  --output_dir experiments/cifar100 \
  --use_nap2
```

ImageNet16-120 (first run downloads the dataset; subsequent runs use cache):

```bash
IMAGENET16_URL=https://your-mirror.example/ImageNet16-120.tar.gz \
python search/evolution_search.py \
  --dataset ImageNet16-120 \
  --search_space micro --init_channels 16 --layers 8 \
  --epochs 20 --pop_size 40 --n_offspring 20 --n_gens 30 \
  --output_dir experiments/imagenet16
```

Quick smoke test (any dataset — replace `--dataset`):

```bash
python search/evolution_search.py \
  --dataset cifar100 \
  --search_space micro --init_channels 16 --layers 8 \
  --epochs 1 --pop_size 2 --n_offspring 2 --n_gens 1 \
  --output_dir experiments/smoke
```

Validation phase (full retraining of a discovered architecture, requires CUDA):

```bash
python validation/train.py \
  --dataset cifar100 \
  --arch NSGANet --layers 20 --init_channels 34 \
  --auxiliary --cutout --batch_size 96 --epochs 600
```

### nap2 predictor checkpoints

When `--use_nap2` is set, the search loads three model files (`.pt`) plus their hyperparameter JSONs. The six paths are configurable in two ways, with CLI taking precedence:

1. **Paste into the constants** at the top of [search/evolution_search.py](search/evolution_search.py): `NAP2_AE_WEIGHTS_PT`, `NAP2_AE_WEIGHTS_JSON`, `NAP2_AE_GRADIENTS_PT`, `NAP2_AE_GRADIENTS_JSON`, `NAP2_LSTM_PT`, `NAP2_LSTM_JSON`.
2. **Pass via CLI flags**: `--nap2_ae_weights_pt`, `--nap2_ae_weights_json`, `--nap2_ae_gradients_pt`, `--nap2_ae_gradients_json`, `--nap2_lstm_pt`, `--nap2_lstm_json`.

The four `.pt` paths and the predictor JSON are required. The two AE JSONs are optional — leave them empty to use the autoencoder's default architecture. Predictor type (LSTM vs BiGRU) and normalization mode are auto-detected from the supplied JSON files.

Example with explicit per-file paths:

```bash
python search/evolution_search.py --use_nap2 \
  --search_space micro --init_channels 16 --layers 8 \
  --epochs 20 --pop_size 40 --n_offspring 20 --n_gens 30 \
  --output_dir experiments/cifar10 \
  --nap2_ae_weights_pt     trained_models/cifar10/ae/weights/ae_weights.pt \
  --nap2_ae_weights_json   trained_models/cifar10/ae/weights/aew_model_hyper_params.json \
  --nap2_ae_gradients_pt   trained_models/cifar10/ae/gradients/ae_gradients.pt \
  --nap2_ae_gradients_json trained_models/cifar10/ae/gradients/aeg_model_hyper_params.json \
  --nap2_lstm_pt           trained_models/cifar10/lstm/cp/model_state_cp/lstm_reg_final.pt \
  --nap2_lstm_json         trained_models/cifar10/lstm/lstm_model_hyper_params.json
```

Without `--use_nap2`, all six paths are ignored — the search runs without the predictor.

## Results on CIFAR-10
![cifar10_pareto](https://github.com/ianwhale/nsga-net/blob/master/img/cifar10.png  "cifar10")

## Pretrained models on CIFAR-10
The easiest way to get started is to evaluate our pretrained NSGA-Net models.

#### Macro search space ([NSGA-Net-macro](https://drive.google.com/file/d/173_CXA_YbEjg1_Lnfg6vqweTRDiuDi0J/view?usp=sharing))
![macro_architecture](https://github.com/ianwhale/nsga-net/blob/beta/img/encoding.png  "architecture")
``` shell
python validation/test.py --net_type macro --model_path weights.pt
```
- Expected result: *3.73%* test error rate with *3.37M* model parameters, *1240M* Multiply-Adds.

#### Micro search space
![micro_architecture](https://github.com/ianwhale/nsga-net/blob/beta/img/cells.png  "Normal&Reduction Cells")
``` shell
python validation/test.py --net_type micro --arch NSGANet --init_channels 26 --filter_increment 4 --SE --auxiliary --model_path weights.pt
```
- Expected result: *2.43%* test error rate with *1.97M* model parameters, *417M* Multiply-Adds ([*weights.pt*](https://drive.google.com/open?id=1JvMkT1eo6JegtUvT-5qY4LK3xgq-k-OH)). 

``` shell
python validation/test.py --net_type micro --arch NSGANet --init_channels 34 --filter_increment 4 --auxiliary --model_path weights.pt
```
- Expected result: *2.22%* test error rate with *2.20M* model parameters, *550M* Multiply-Adds ([*weights.pt*](https://drive.google.com/open?id=1it_aFoez-U7SkxSuRPYWDVFg8kZwE7E7)). 

``` shell
python validation/test.py --net_type micro --arch NSGANet --init_channels 36 --filter_increment 6 --SE --auxiliary --model_path weights.pt
```
- Expected result: *2.02%* test error rate with *4.05M* model parameters, *817M* Multiply-Adds ([*weights.pt*](https://drive.google.com/open?id=1kLXzKxQ7dazjmANTvgSoeMPHWwYKiOtm)). 

## Pretrained models on CIFAR-100
``` shell
python validation/test.py --task cifar100 --net_type micro --arch NSGANet --init_channels 36 --filter_increment 6 --SE --auxiliary --model_path weights.pt
```
- Expected result: *14.42%* test error rate with *4.1M* model parameters, *817M* Multiply-Adds ([*weights.pt*](https://drive.google.com/open?id=1CMtSg1l2V5p0HcRxtBsD8syayTtS9QAu)). 

## Architecture validation
To validate the results by training from scratch, run
``` 
# architecture found from macro search space
python validation/train.py --net_type macro --cutout --batch_size 128 --epochs 350 
# architecture found from micro search space
python validation/train.py --net_type micro --arch NSGANet --layers 20 --init_channels 34 --filter_increment 4  --cutout --auxiliary --batch_size 96 --droprate 0.2 --SE --epochs 600
```
You may need to adjust the batch_size depending on your GPU memory. 

For customized macro search space architectures, change `genome` and `channels` option in `train.py`. 

For customized micro search space architectures, specify your architecture in `models/micro_genotypes.py` and use `--arch` flag to pass the name. 


## Architecture search 
To run architecture search:
``` shell
# macro search space
python search/evolution_search.py --search_space macro --init_channels 32 --n_gens 30
# micro search space
python search/evolution_search.py --search_space micro --init_channels 16 --layers 8 --epochs 20 --n_offspring 20 --n_gens 30
```
Pareto Front               |  Network                  
:-------------------------:|:-------------------------:
![](https://github.com/ianwhale/nsga-net/blob/beta/img/pf_macro.gif)  |  ![](https://github.com/ianwhale/nsga-net/blob/beta/img/macro_network.gif)

Pareto Front               |  Normal Cell              | Reduction Cell
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/ianwhale/nsga-net/blob/beta/img/pf_micro.gif)  |  ![](https://github.com/ianwhale/nsga-net/blob/beta/img/nd_normal_cell.gif)  |  ![](https://github.com/ianwhale/nsga-net/blob/beta/img/nd_reduce_cell.gif)

If you would like to run asynchronous and parallelize each architecture's back-propagation training, set `--n_offspring` to `1`. The algorithm will run in *steady-state* mode, in which the population is updated as soon as one new architecture candidate is evaludated. It works reasonably well in single-objective case, a similar strategy is used in [here](https://arxiv.org/abs/1802.01548).  

## Visualization
To visualize the architectures:
``` shell
python visualization/macro_visualize.py NSGANet            # macro search space architectures
python visualization/micro_visualize.py NSGANet            # micro search space architectures
```
For customized architecture, first define the architecture in `models/*_genotypes.py`, then substitute `NSGANet` with the name of your customized architecture. 

## Citations
If you find the code useful for your research, please consider citing our works
``` 
@article{nsganet,
  title={NSGA-NET: a multi-objective genetic algorithm for neural architecture search},
  author={Lu, Zhichao and Whalen, Ian and Boddeti, Vishnu and Dhebar, Yashesh and Deb, Kalyanmoy and Goodman, Erik and  Banzhaf, Wolfgang},
  booktitle={GECCO-2019},
  year={2018}
}
```

## Acknowledgement 
Code heavily inspired and modified from [pymoo](https://github.com/msu-coinlab/pymoo), [DARTS](https://github.com/quark0/darts#requirements) and [pytorch-cifar10](https://github.com/kuangliu/pytorch-cifar). 
