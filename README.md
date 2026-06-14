# FoCuS-SR

Official code release for **FoCuS-SR**, a diffusion-prior real-world image super-resolution method built around function-aware curriculum learning and safe-to-risk layer-wise semantic LoRA training.

## Method Pipeline

FoCuS-SR trains a Stable-Diffusion-based SR model in two functional stages:

1. **Stage1 restoration foundation**: train pixel LoRA on all training images with L2 loss and online broad Real-ESRGAN degradation presets.
2. **Stage2 data curriculum**: select high-quality and semantically diverse training images using MUSIQ quality filtering and CLIP image-feature clustering.
3. **Merged-weight layer probing**: merge the Stage1 pixel LoRA into the base UNet, then probe semantic-LoRA-eligible layers for importance and conflict.
4. **Safe-to-risk semantic curriculum**: train low-conflict semantic layers first, then high-conflict semantic layers under stronger fidelity and consistency constraints.
5. **Importance-based rank allocation**: assign semantic LoRA rank from layer importance while keeping all semantic layers in the model.

## Installation

Create an environment and install dependencies:

```bash
conda create -n focussr python=3.10 -y
conda activate focussr
pip install -r requirements.txt
```

If your machine needs a custom PyTorch/CUDA build, install PyTorch first from the official PyTorch instructions, then install the remaining packages from `requirements.txt`.

## Prepare Assets

Large assets are not included in this repository. Prepare them locally with the following layout:

```text
preset/
  models/
    stable-diffusion-2-1-base/
    focussr.pkl                 # optional pretrained FoCuS-SR checkpoint for inference
  test_datasets/
  testfolder/
    realsr/
      test_LR/
      test_HR/
    drealsr/
      test_LR/
      test_HR/
    div2k/
      lq/
      gt/
src/
  ram_pretrain_model/
    ram_swin_large_14m.pth
```

### Model Weights

Install the Hugging Face Hub CLI if needed:

```bash
pip install -U huggingface_hub
```

Optional mirror setting for regions where Hugging Face is slow:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Download Stable Diffusion 2.1 base in Diffusers format:

```bash
mkdir -p preset/models
huggingface-cli download stabilityai/stable-diffusion-2-1-base \
  --local-dir preset/models/stable-diffusion-2-1-base \
  --local-dir-use-symlinks False
```

If the official SD2.1 repository is unavailable in your region, use any licensed Diffusers-compatible copy of `stable-diffusion-2-1-base` and place it at:

```text
preset/models/stable-diffusion-2-1-base/
```

Download the RAM checkpoint used by the prompt/tagging module:

```bash
mkdir -p src/ram_pretrain_model
wget -O src/ram_pretrain_model/ram_swin_large_14m.pth \
  https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth
```

For inference with a released FoCuS-SR checkpoint, place it at:

```text
preset/models/focussr.pkl
```

If no pretrained checkpoint is available, train the model with the command in the [Training](#training) section and use the final Stage2b checkpoint for inference/evaluation.

### Training Datasets

FoCuS-SR uses HR image path lists as training input. We follow the common PiSA-SR/OSEDiff-style setting with LSDIR and FFHQ, but any local HR image collection can be used if the path lists are prepared correctly.

Recommended sources:

- LSDIR: obtain from the official NTIRE/LSDIR release page or challenge dataset distribution.
- FFHQ: obtain from the official NVIDIA FFHQ dataset repository: https://github.com/NVlabs/ffhq-dataset

Example local layout:

```text
/data/LSDIR/HR/
/data/FFHQ/
```

Generate path lists:

```bash
find /data/LSDIR/HR -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort > preset/gt_lsdir_path.txt
find /data/FFHQ -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort > preset/gt_ffhq_path.txt
cat preset/gt_lsdir_path.txt preset/gt_ffhq_path.txt > preset/gt_all_path.txt
```

Stage2 HQ data can be generated automatically by:

```bash
GPU=0 RUN_NAME=focus_sr_main bash scripts/train_focussr.sh
```

The training script writes the selected HQ list to:

```text
experiments/focus_sr_main/data/gt_hq_clip_musiq_path.txt
```

If you want to skip data selection, prepare `preset/gt_hq_path.txt` manually and run training with:

```bash
RUN_DATA_SELECTION=0 GPU=0 RUN_NAME=focus_sr_main bash scripts/train_focussr.sh
```

### Evaluation Benchmarks

For RealSR evaluation, download the official RealSR benchmark from:

```text
https://github.com/csjcai/RealSR
```

Place LR and HR test images as:

```text
preset/testfolder/realsr/test_LR/
preset/testfolder/realsr/test_HR/
preset/testfolder/drealsr/test_LR/
preset/testfolder/drealsr/test_HR/
preset/testfolder/div2k/lq/
preset/testfolder/div2k/gt/
```

or pass explicit paths when running evaluation:

```bash
DATASET=realsr INPUT=/path/to/RealSR/test_LR GT=/path/to/RealSR/test_HR \
GPU=0 CKPT=<path_to_checkpoint.pkl> EXP_NAME=focus_sr_main EVAL_NAME=realsr \
bash scripts/eval.sh
```

MUSIQ and CLIP/OpenCLIP weights used by `scripts/build_data_curriculum.py` are downloaded to the local cache automatically by their Python packages on first use. If the machine has no internet access, pre-cache those weights before running data selection.

## Prepare Data Lists

If you do not want to use the `find` commands above, start from the example files:

```bash
cp preset/gt_all_path.example.txt preset/gt_all_path.txt
cp preset/gt_hq_path.example.txt preset/gt_hq_path.txt
cp preset/gt_lsdir_path.example.txt preset/gt_lsdir_path.txt
cp preset/gt_ffhq_path.example.txt preset/gt_ffhq_path.txt
```

Then edit each `preset/gt_*_path.txt` file so every line points to one local HR image. `preset/gt_all_path.txt` is used by Stage1. `preset/gt_hq_path.txt` is only required when running with `RUN_DATA_SELECTION=0`.

## Training

Run the full FoCuS-SR main pipeline:

```bash
GPU=0 RUN_NAME=focus_sr_main bash scripts/train_focussr.sh
```

Useful smoke test:

```bash
GPU=0 PROBE_BATCHES=1 STAGE1_STEPS=1 STAGE2A_STEPS=1 STAGE2B_STEPS=1 \
RUN_DATA_SELECTION=0 RUN_NAME=smoke \
bash scripts/train_focussr.sh
```

Important defaults:

- Stage1 uses `preset/gt_all_path.txt`.
- Stage1 degradation presets are `src/datasets/params_stage1_mild.yml`, `src/datasets/params_stage1_medium.yml`, and `src/datasets/params_stage1_heavy.yml`.
- Stage2 HQ data is generated by `scripts/build_data_curriculum.py` unless `RUN_DATA_SELECTION=0`.
- Stage2a trains low-conflict semantic modules.
- Stage2b trains high-conflict semantic modules.
- Outputs are written to `experiments/${RUN_NAME}/`.

## Inference

Run inference on a folder or image:

```bash
GPU=0 CKPT=preset/models/focussr.pkl \
bash scripts/infer.sh preset/test_datasets results/demo
```

The command writes SR images to `results/demo`.

## Evaluation

Run RealSR, DRealSR, and DIV2K inference and metrics:

```bash
GPU=0 CKPT=<path_to_checkpoint.pkl> EXP_NAME=focus_sr_main EVAL_NAME=s8500 \
bash scripts/eval_all.sh
```

Run a single dataset:

```bash
DATASET=realsr GPU=0 CKPT=<path_to_checkpoint.pkl> EXP_NAME=focus_sr_main EVAL_NAME=s8500 \
bash scripts/eval.sh
```

Supported `DATASET` values are `realsr`, `drealsr`, and `div2k`. By default, the scripts read:

```text
realsr:  INPUT=preset/testfolder/realsr/test_LR,  GT=preset/testfolder/realsr/test_HR
drealsr: INPUT=preset/testfolder/drealsr/test_LR, GT=preset/testfolder/drealsr/test_HR
div2k:   INPUT=preset/testfolder/div2k/lq,        GT=preset/testfolder/div2k/gt
```

Override them when evaluating another benchmark:

```bash
DATASET=realsr INPUT=/path/to/RealSR/test_LR GT=/path/to/RealSR/test_HR \
GPU=0 CKPT=<path_to_checkpoint.pkl> EXP_NAME=focus_sr_main EVAL_NAME=custom \
bash scripts/eval.sh
```

## Repository Layout

```text
model.py                         # FoCuS-SR model and LoRA setup
train.py                         # training loop
infer.py                         # inference entrypoint
probe.py                         # layer probing entrypoint
metrics.py                       # local metric evaluation
scripts/train_focussr.sh          # full main-method training pipeline
scripts/probe_layers.sh           # merged-weight semantic layer probing
scripts/build_layer_curriculum.py # safe-to-risk split and rank curriculum builder
scripts/build_data_curriculum.py  # MUSIQ + CLIP HQ data curriculum builder
scripts/infer.sh                  # simple inference wrapper
scripts/eval.sh                   # single-dataset inference + metrics wrapper
scripts/eval_all.sh               # RealSR + DRealSR + DIV2K evaluation wrapper
```

## Citation and Acknowledgement

Citation will be added after release.

This codebase builds on ideas and public implementations from PiSA-SR, SeeSR, OSEDiff, Stable Diffusion, Real-ESRGAN, RAM, and related real-world super-resolution projects. We thank the authors for making their work available.
