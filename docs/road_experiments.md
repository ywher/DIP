# Road-scene DIP reproduction

## Status of the official release

The original entry points are `trainDIP.py`, `getsupp_pro.py`, and
`testDIP.py`. They implement the published support/query prototype pipeline,
but dataset paths, task-specific dataloaders, saved prototype names, and test
paths are hard-coded. Only GTA5-to-Cityscapes, SYNTHIA-to-Cityscapes, and
Cityscapes-to-BDD100K are partially represented. The released commands are
therefore a useful algorithm reference, but are not directly executable as a
five-transfer benchmark.

The nominal official sequence is:

```bash
python trainDIP.py --config config/pascal/pascal_split0_resnet50.yaml
python getsupp_pro.py --config config/pascal/pascal_split0_resnet50.yaml
python testDIP.py --config config/pascal/pascal_split0_resnet50.yaml
```

It is not an executable reproduction recipe without source edits: the checked-in
training script currently activates `City2BDD`, the YAML filename says
ResNet-50 while `TRAIN.layers` is 101, and prototype/checkpoint paths are fixed
independently in all three scripts. Use this sequence only to inspect the
official implementation; use `tools/road/run.py` for controlled experiments.

The new `road_dip` harness leaves those files unchanged and provides:

- five manifest-based driving transfers;
- the published ResNet-101 prototype encoder (`native`);
- a standalone DINOv3-B + ReIN + HRDA dense embedding encoder (`vfm`);
- the original DIP masked-average prototypes and cosine classifier;
- checkpointing, prototype extraction, validation, mIoU, ETA, and resume;
- matched image-level target support lists at 1/64 (1/128 for Mapillary).

## Protocol boundary

DIP is an adaptation method, not an acquisition method. Its official
Cityscapes one-/five-shot files are manually assembled support sets. The road
configs therefore use the same labeled target-image manifests as TC-ADA so
that the experiment isolates the adaptation model under a matched annotation
set. Report these rows as **DIP with a matched image split**, not as a native
DIP acquisition result.

The VFM profile preserves DIP's training objective. DINOv3-B + ReIN + HRDA
produces a dense 256-D embedding; target support masks form class prototypes,
and source pixels are classified by cosine distance to those prototypes. HRDA
uses a scalar low/high-resolution fusion gate because class-specific gates do
not exist before prototypes are constructed.

## Setup

```bash
conda activate reinpy10
cd third_party/DIP-hunnu
bash scripts/setup_road_data.sh /path/to/datasets

mkdir -p pretrained/dinov3
ln -s /path/to/resnet101_v2.pth pretrained/resnet101_v2.pth
ln -s /path/to/dinov3_vitb16.pth pretrained/dinov3/dinov3_vitb16.pth
```

Check all ten profiles before training:

```bash
python tools/road/check_setup.py configs/road/{native,vfm}/*.yaml
```

## Train and evaluate

```bash
# Published ResNet-101 encoder
bash scripts/run_road_experiment.sh \
  configs/road/native/gta2cityscapes_1_64.yaml 0

# DINOv3-B + ReIN + HRDA encoder
bash scripts/run_road_experiment.sh \
  configs/road/vfm/gta2cityscapes_1_64.yaml 0

# Resume
bash scripts/run_road_experiment.sh CONFIG GPU outputs/.../checkpoint_020000.pth

# Recompute validation for selected checkpoints
bash scripts/eval_road_checkpoints.sh CONFIG GPU \
  outputs/.../checkpoint_010000.pth outputs/.../checkpoint_040000.pth
```

Run all five tasks sequentially on one GPU:

```bash
bash scripts/run_five_road_tasks.sh native 0
bash scripts/run_five_road_tasks.sh vfm 0
```

The default road protocol uses 40k iterations and one source--support pair
(two images) per step, with checkpoints and validation every 10k iterations
and the same 1024x1024 crop used by the controlled VFM comparisons. Training
uses FP32: FP16 gradient scaling overflows in the cosine-prototype branch and
can silently skip optimizer updates.
