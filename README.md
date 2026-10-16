# AttributeIR: Visual Attribute-Driven All-in-One Image Restoration

This repository is a clean, reproducible implementation of **AttributeIR**, following the paper *AttributeIR: Visual Attribute-Driven All-in-One Image Restoration*. The project contains the restoration model, grouped Attribute Adjustment Modules (AAMs), paired dataset loaders, Q-Instruct attribute-context generation, training, evaluation, checkpoint validation, and metric reporting.

## What the method does

AttributeIR has two stages:

1. A frozen Q-Instruct vision-language IQA model queries eight restoration-relevant attributes: `Brightness`, `Contrast`, `Color`, `Sharpness`, `Structure`, `Texture`, `Noise`, and `Artifact`. The hidden state at the score position is stored as a 4096-dimensional attribute representation, together with the nine preference logits.
2. A four-level Restormer-style U-Net uses the representations as restoration guidance. Each decoder level contains an AAM with three parallel groups:
   - Photometric: brightness, contrast, color;
   - Fidelity: sharpness, structure, texture;
   - Corruption: noise, artifact.

The implementation uses visual priors from the degraded image for spatial evidence and residual reconstruction. The training objective is reconstruction supervision plus an attribute-transition term. The code also includes conservative branch regularizers used by the released checkpoint to keep the three adapters stable during optimization.

## Paper-aligned defaults

The default training configuration follows the paper: encoder blocks `[4, 6, 6, 8]`, decoder blocks `[6, 6, 4]`, AAM at each decoder level, random `128 x 128` crops, horizontal flips, Adam with learning rate `5e-4`, cosine annealing, batch size `32`, and `L = L1 + 0.2 L_attr`. The Q-Instruct model is frozen and its key-value cache is reused across attribute queries.

The code supports the paper's five ID tasks: Rain100L, SOTS, BSD68 sigma=25, LOL, and GoPro. It also provides loaders for the paper's OOD tasks and the additional dataset layouts present in the accompanying dataset tree.

## Repository layout

```text
AttributeIR_paper_release/
├── models/
│   ├── attribute_ir.py          # Restormer-style AttributeIR network
│   └── attribute_adapters.py    # Photometric/Fidelity/Corruption AAMs and losses
├── data/
│   ├── paired_dataset.py        # training pairs, balancing, CSV context loading
│   └── test_dataset.py          # explicit ID/OOD test pairs
├── tools/
│   ├── generate_attribute_context.py  # Q-Instruct CSV generation
│   ├── q_instruct_online.py            # online context extractor for evaluation
│   └── checkpoint_check.py              # strict load and forward smoke test
├── train.py
├── test.py
├── configs/paper.yaml
├── check/                         # put a checkpoint here; large weights use Git LFS
└── utils/image_utils.py
```

## Installation

```bash
conda create -n attributeir python=3.10 -y
conda activate attributeir
pip install -r requirements.txt
```

Install the Q-Instruct/mPLUG-Owl2 implementation and its model weights separately. The context generator expects the model identifier or local model directory accepted by `load_pretrained_model`.

## Training

From the repository root:

```bash
python train.py \
  --dataset-root /path/to/Baseline/datasets \
  --tasks derain,dehaze,deblur,light,noise \
  --epochs 10 \
  --batch-size 32 \
  --crop-size 128 \
  --lr 5e-4 \
  --checkpoint-dir ./checkpoints \
  --device cuda --amp
```

The training loader pairs input and target images by basename, balances the five main tasks, loads degraded and clean attribute contexts, and applies the same spatial crop and flip to both images. For a small smoke test, use `--target-per-task 4 --epochs 1 --batch-size 1 --num-workers 0`.

## Testing and metrics

The test entry point supports paired ID/OOD datasets, padding to a multiple of 64 for the U-Net, image export, PSNR, SSIM, and a CSV summary:

```bash
python test.py \
  --checkpoint ./check/combined_5d_backbone_aam.pth \
  --dataset-root your_path \
  --test-size 0 \
  --output-dir ./results/attribute_ir \
  --device cuda
```

`--test-size 0` keeps the complete original image during restoration. The
network only pads the tensor to a multiple of 64 and removes that padding
afterward; it does not evaluate a center patch. If a bounded center crop is
needed for a separate experiment, pass a positive `--test-size` explicitly.

For datasets without paired clean references, use the corresponding task only for image export or integrate no-reference metrics separately. The paper reports PSNR/SSIM for paired datasets and MUSIQ/MANIQA for perceptual evaluation.

Validate a checkpoint before a long test run:

```bash
python tools/checkpoint_check.py \
  --checkpoint ./check/combined_5d_backbone_aam.pth \
  --device cuda --image-size 128 --strict
```

