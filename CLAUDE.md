# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the official research code for the paper "Enhancing Zero-Shot Facial Expression
Recognition by LLM Knowledge Transfer" (Exp-CLIP, WACV 2025). It trains a small projection
head on top of a frozen CLIP vision-language model so that CLIP's image embeddings align with
facial-action semantics distilled from a large language model (BLIP-2 + FlanT5-XXL), then
evaluates the result via zero-shot classification on seven in-the-wild facial expression
recognition (FER) datasets. This is a research/experiment repo, not a library or service —
there is no test suite, linter, or CI configured.

## Environment Setup

There is no proper `pip install -r requirements.txt` — `requirements.txt` is actually a list of
setup commands to run manually:

```bash
conda create -n ExpCLIP python=3.10 -y
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install transformers tqdm regex pyyaml pandas pydantic click
pip install timm einops
```

Additionally, `models/BLIP2_T5.py` imports `lavis` (the Salesforce LAVIS library, for
`load_model_and_preprocess("blip2_t5", "pretrain_flant5xxl", ...)`) which is **not** listed in
`requirements.txt` but is required to run training (it loads the frozen BLIP-2/FlanT5-XXL
teacher model). It is only needed for training, not for zero-shot testing.

`models/clip/` is a vendored copy of OpenAI's CLIP implementation (not pip-installed); it
downloads CLIP weights on first `clip.load(...)` call.

## Running Training and Evaluation

Before running anything, two hardcoded absolute dataset paths must be edited for your machine:
- `train.py`: `train_data_file_path` (an `ImageFolder`-style directory of unlabelled face images
  used to train the projection head, e.g. CAER-S faces)
- `test.py`: `DATASET_PATH_MAPPING` dict (paths to RAFDB/AffectNet7/AffectNet8/FERPlus image
  folders; the dynamic (video) datasets DFEW/FERV39k/MAFW/AFEW instead read from the `.txt`
  list files already checked into `./annotation/`, which contain absolute frame-directory paths
  that also need updating for your environment — see the README's link to DFER-CLIP's
  preprocessing docs)

Standard end-to-end run (train projection head, then run all zero-shot evals):

```bash
sh runner.sh
```

This runs `train.py` then `test.py` with a shared `$JOB_ID` (must be exported/set beforehand),
e.g.:

```bash
JOB_ID=myrun sh runner.sh
```

Key `train.py` flags (see `runner.sh` for the reference invocation): `--load-model` selects the
frozen CLIP backbone (`CLIP_B32` | `CLIP_B16` | `CLIP_L14`), `--instruction` is the text prompt
sent to the BLIP-2/T5 teacher to elicit facial-action descriptions, `--job-id` names the
log/checkpoint output, `--milestones` are LR-decay epochs for `MultiStepLR`.

`test.py --load-model <name> --job-id <id>` loads `./checkpoint/<job-id>-model.pth`, runs
zero-shot evaluation across all 5 prompt-template types (`FER_prompt_list` in
`models/Text.py`) on both static (image-folder) and dynamic (video-frame) datasets, prints
UAR/WAR per dataset, and writes confusion-matrix PDFs to `./confusion_matrix/`.

Logs go to `./log/<job-id>-log.txt` (text) and `./log/<job-id>-log.png` (accuracy/loss curve,
via `RecorderMeter` in `train.py`). Example logs/outputs from the paper's runs are checked in as
`ExpCLIP_B16.txt`, `ExpCLIP_B32.txt`, `ExpCLIP_L14.txt` and under `./log/`.

Pretrained projection-head checkpoints for all three backbones are in `./checkpoint/` — you can
run `test.py` directly against these without training first.

## Architecture

**Core idea**: CLIP's image/text encoders are frozen; the only trained component is a single
linear layer (`projection_head`) that maps CLIP's image-embedding space into the embedding space
of a frozen LLM (BLIP-2's FlanT5-XXL text encoder), which is assumed to better capture
fine-grained facial-action semantics. Training uses a symmetric InfoNCE / CLIP-style contrastive
loss (`train.py:train()`) between projected image features and LLM text features describing that
same image.

- **`models/Exp_CLIP.py`** — the two top-level model wrappers:
  - `ExpCLIP_Train`: holds the frozen CLIP backbone, the frozen BLIP-2 teacher (`get_blip2t5_model`),
    and the trainable `projection_head` (`Linear_Matrix` for B32/B16 CLIP, 512→4096;
    `Linear_Matrix_L14` for L14 CLIP, 768→4096, matching FlanT5-XXL's hidden size). `forward()`
    runs an image through both CLIP (→ projected image features) and BLIP-2 (→ mean-pooled T5
    encoder features), both L2-normalized.
  - `ExpCLIP_Test`: frozen CLIP + `projection_head` only (no BLIP-2 — the teacher is not needed at
    test time). `forward()` supports `mode_task="Static_FER"` (single image) or
    `"Dynamic_FER"` (multi-frame video clips, averaged over the time dimension). Text prompts are
    tokenized with CLIP's own tokenizer and pushed through the *same* projection head, so
    image and text end up compared in the same projected space at test time.
  - In `train.py`, only parameters whose name contains `"projection_head"` have `requires_grad`
    enabled — everything else (CLIP, BLIP-2) stays frozen.

- **`models/BLIP2_T5.py`** — wraps LAVIS's `blip2_t5` model. The key trick: it monkey-patches the
  model's `.generate` method (via `MethodType`) to register a forward hook on
  `t5_model.encoder.final_layer_norm` and run T5's `.generate()` for its side effect — capturing
  the *encoder's* hidden states (via the hook) rather than using the decoded text output. This is
  how the LLM's semantic representation of "please describe the facial actions in this image" is
  extracted for the contrastive loss (see `get_tokens`/`get_blip2t5_model`, used from
  `ExpCLIP_Train.__init__`/`forward`). The instruction text is fixed per training run (`--instruction`)
  and pre-tokenized once for the whole batch size in `__init__`.

- **`models/clip/`** — vendored OpenAI CLIP (`clip.py`, `model.py`, `simple_tokenizer.py`); use
  `clip.load("ViT-B/32" | "ViT-B/16" | "ViT-L/14", device)` and `clip.tokenize(...)`.

- **`models/Text.py`** — all zero-shot prompt templates and per-dataset class-name lists.
  `FER_prompt_list` has 5 prompt "types" (class name only, "an expression of X", "a photo of a
  face with an expression of X", and two multi-template ensembles of 5/10 phrasings per class),
  each keyed by dataset name. `Emotion_name_dic` gives the ordered class names per dataset (for
  confusion-matrix axis labels). Datasets vary in class count: 7 classes for
  RAFDB/AffectNet7/DFEW/FERV39k/AFEW, 8 for AffectNet8/FERPlus, 11 for MAFW. `test.py` divides
  by these counts to know how many prompt variants to average per class when using ensemble
  prompts.

- **`data_loader/video_dataloader.py`** — `VideoDataset` reads a list file of
  `<frame_dir> <num_frames> <label>` lines (see `./annotation/*.txt`), and samples
  `num_segments` frames per video (uniform-random per segment for train, mid-segment for test)
  from `glob.glob(<frame_dir>/*)` (pre-extracted face-cropped frames, not raw video). Returns a
  `(num_segments, 3, H, W)`-shaped tensor per clip that `ExpCLIP_Test`/training code then
  reshapes to run per-frame CLIP encoding before averaging over time.

- **`data_loader/video_transform.py`** — group-wise (multi-frame-consistent) versions of
  standard torchvision transforms (`GroupRandomSizedCrop`, `GroupRandomHorizontalFlip`,
  `GroupResize`, `Stack`, `ToTorchFormatTensor`, etc.) so the same crop/flip is applied
  consistently across all frames of a clip.

## Datasets

- **Static (image-folder) FER**: RAFDB, AffectNet7, AffectNet8, FERPlus — loaded via
  `torchvision.datasets.ImageFolder`, evaluated with `mode_task="Static_FER"`.
- **Dynamic (video-frame) FER**: DFEW (5 CV folds), FERV39k (1 split), MAFW (5 CV folds), AFEW
  (1 split) — loaded via `VideoDataset` from the `.txt` list files in `./annotation/`, evaluated
  with `mode_task="Dynamic_FER"` and averaged over `num_segments=16` frames per clip. `test.py`
  averages UAR/WAR across folds for datasets with multiple CV splits.
- The unlabelled training data (`train_data_file_path` in `train.py`) is a plain face-image
  folder (paper uses CAER-S) — labels aren't used during training, only the image itself (both
  the CLIP branch and the BLIP-2 branch consume the same image).
