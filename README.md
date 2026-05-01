# Graph-CoOp: Setup and Run Instructions

## Prerequisites

- GPU with CUDA (tested on: tell them your target GPU)
- Conda

## Setup (do this once)

```bash
git clone https://github.com/KaiyangZhou/CoOp.git
cd CoOp

conda create -n coop python=3.8 -y
conda activate coop

pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install ftfy regex tqdm

pip install git+https://github.com/openai/CLIP.git

git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch && python setup.py develop && cd ..
```

## Add our files

Copy the received files into the CoOp directory,
preserving the folder structure exactly as sent.

## Download datasets

Follow CoOp/DATASETS.md — at minimum download:

- Caltech101
- EuroSAT
- Flowers102

## Verify tests pass (optional but recommended)

```bash
python test_graph_utils.py
python test_gcn.py
python test_forward_backward.py
# All 7 tests should print PASS
```

## Run experiments

### 1. Baseline (original CoOp — reproduce paper numbers first)

```bash
bash scripts/coop/main.sh caltech101 rn50 end 16 16 False
# Expected accuracy: ~91.8%
```

### 2. Graph Label Smoothing only

```bash
python train.py \
  --root /path/to/datasets \
  --trainer CoOpGraph \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --config-file configs/trainers/CoOp/rn50_ep50_graph.yaml \
  --output-dir output/graph/caltech101/smooth_only/16shots \
  DATASET.NUM_SHOTS 16 \
  TRAINER.COOP.ALPHA_SMOOTH 0.1 \
  TRAINER.COOP.LAMBDA_LAP 0.0 \
  TRAINER.COOP.USE_GCN False
```

### 3. Laplacian regularization only

```bash
python train.py \
  --root /path/to/datasets \
  --trainer CoOpGraph \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --config-file configs/trainers/CoOp/rn50_ep50_graph.yaml \
  --output-dir output/graph/caltech101/lap_only/16shots \
  DATASET.NUM_SHOTS 16 \
  TRAINER.COOP.ALPHA_SMOOTH 0.0 \
  TRAINER.COOP.LAMBDA_LAP 0.01 \
  TRAINER.COOP.USE_GCN False
```

### 4. Smoothing + Laplacian combined

```bash
python train.py \
  --root /path/to/datasets \
  --trainer CoOpGraph \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --config-file configs/trainers/CoOp/rn50_ep50_graph.yaml \
  --output-dir output/graph/caltech101/smooth_lap/16shots \
  DATASET.NUM_SHOTS 16 \
  TRAINER.COOP.ALPHA_SMOOTH 0.1 \
  TRAINER.COOP.LAMBDA_LAP 0.01 \
  TRAINER.COOP.USE_GCN False
```

### 5. Full graph (with GCN)

```bash
python train.py \
  --root /path/to/datasets \
  --trainer CoOpGraph \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --config-file configs/trainers/CoOp/rn50_ep50_graph.yaml \
  --output-dir output/graph/caltech101/full_graph/16shots \
  DATASET.NUM_SHOTS 16 \
  TRAINER.COOP.ALPHA_SMOOTH 0.1 \
  TRAINER.COOP.LAMBDA_LAP 0.01 \
  TRAINER.COOP.USE_GCN True
```

## Run all shot counts (for plotting curves like paper Fig 3)

```bash
for SHOTS in 1 2 4 8 16; do
  python train.py \
    --root /path/to/datasets \
    --trainer CoOpGraph \
    --dataset-config-file configs/datasets/eurosat.yaml \
    --config-file configs/trainers/CoOp/rn50_ep50_graph.yaml \
    --output-dir output/graph/eurosat/smooth_lap/${SHOTS}shots \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.COOP.ALPHA_SMOOTH 0.1 \
    TRAINER.COOP.LAMBDA_LAP 0.01 \
    TRAINER.COOP.USE_GCN False
done
```

## Expected results to report back

For each experiment, share:

1. Final test accuracy (printed at end of log)
2. The output log file from output/.../<exp_name>/log.txt

## Recommended run order

1. caltech101, 16 shots, all 5 configs (fast, ~15 min each on GPU)
2. eurosat, all shot counts, smooth_lap config
3. flowers102, all shot counts, smooth_lap config

# Prompt Learning for Vision-Language Models

This repo contains the codebase of a series of research projects focused on adapting vision-language models like [CLIP](https://arxiv.org/abs/2103.00020) to downstream datasets via _prompt learning_:

- [Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557), in CVPR, 2022.
- [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134), IJCV, 2022.

## Updates

- **07.10.2022**: Just added to both [CoOp](https://arxiv.org/abs/2109.01134) and [CoCoOp](https://arxiv.org/abs/2203.05557) (in their appendices) the results on the newly proposed DOSCO (DOmain Shift in COntext) benchmark, which focuses on contextual domain shift and covers a diverse set of classification problems. (The paper about DOSCO is [here](https://arxiv.org/abs/2209.07521) and the code for running CoOp/CoCoOp on DOSCO is [here](https://github.com/KaiyangZhou/on-device-dg).)

- **17.09.2022**: [Call for Papers](https://kaiyangzhou.github.io/assets/cfp_ijcv_lvms.html): IJCV Special Issue on _The Promises and Dangers of Large Vision Models_.

- **16.07.2022**: CoOp has been accepted to IJCV for publication!

- **10.06.2022**: Our latest work, [Neural Prompt Search](https://arxiv.org/abs/2206.04673), has just been released on arxiv. It provides a novel perspective for fine-tuning large vision models like [ViT](https://arxiv.org/abs/2010.11929), so please check it out if you're interested in parameter-efficient fine-tuning/transfer learning. The code is also made public [here](https://github.com/Davidzhangyuanhan/NOAH).

- **08.06.2022**: If you're looking for the code to draw the few-shot performance curves (like the ones we show in the CoOp's paper), see `draw_curves.py`.

- **09.04.2022**: The pre-trained weights of CoOp on ImageNet are released [here](#pre-trained-models).

- **11.03.2022**: The code of our CVPR'22 paper, "[Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557)," is released.

- **15.10.2021**: We find that the `best_val` model and the `last_step` model achieve similar performance, so we set `TEST.FINAL_MODEL = "last_step"` for all datasets to save training time. Why we used `best_val`: the ([tiny](https://github.com/KaiyangZhou/CoOp/blob/main/datasets/oxford_pets.py#L32)) validation set was designed for the linear probe approach, which requires extensive tuning for its hyperparameters, so we used the `best_val` model for CoOp as well for fair comparison (in this way, both approaches have access to the validation set).

- **09.10.2021**: Important changes are made to Dassl's transforms.py. Please pull the latest commits from https://github.com/KaiyangZhou/Dassl.pytorch and this repo to make sure the code works properly. In particular, 1) `center_crop` now becomes a default transform in testing (applied after resizing the smaller edge to a certain size to keep the image aspect ratio), and 2) for training, `Resize(cfg.INPUT.SIZE)` is deactivated when `random_crop` or `random_resized_crop` is used. Please read this [issue](https://github.com/KaiyangZhou/CoOp/issues/8) on how these changes might affect the performance.

- **18.09.2021**: We have fixed an error in Dassl which could cause a training data loader to have zero length (so no training will be performed) when the dataset size is smaller than the batch size (due to `drop_last=True`). Please pull the latest commit for Dassl (>= `8eecc3c`). This error led to lower results for CoOp in EuroSAT's 1- and 2-shot settings (others are all correct). We will update the paper on arxiv to fix this error.

## How to Install

This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `CoOp/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.

## How to Run

Click a paper below to see the detailed instructions on how to run the code to reproduce the results.

- [Learning to Prompt for Vision-Language Models](COOP.md)
- [Conditional Prompt Learning for Vision-Language Models](COCOOP.md)

## Models and Results

- The pre-trained weights of CoOp (both M=16 & M=4) on ImageNet based on RN50, RN101, ViT-B/16 and ViT-B/32 can be downloaded altogether via this [link](https://drive.google.com/file/d/18ypxfd82RR0pizc5MM1ZWDYDk4j0BtPF/view?usp=sharing). The weights can be used to reproduce the results in Table 1 of CoOp's paper (i.e., the results on ImageNet and its four variants with domain shift). To load the weights and run the evaluation code, you will need to specify `--model-dir` and `--load-epoch` (see this [script](https://github.com/KaiyangZhou/CoOp/blob/main/scripts/eval.sh) for example).
- The raw numerical results can be found at this [google drive link](https://docs.google.com/spreadsheets/d/12_kaFdD0nct9aUIrDoreY0qDunQ9q9tv/edit?usp=sharing&ouid=100312610418109826457&rtpof=true&sd=true).

## Citation

If you use this code in your research, please kindly cite the following papers

```bash
@inproceedings{zhou2022cocoop,
    title={Conditional Prompt Learning for Vision-Language Models},
    author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}

@article{zhou2022coop,
    title={Learning to Prompt for Vision-Language Models},
    author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
    journal={International Journal of Computer Vision (IJCV)},
    year={2022}
}
```
