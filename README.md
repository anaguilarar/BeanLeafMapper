# BeanLeafMapper

SAM3-based leaf detection and leaf-area estimation for bean crops.

## What it does

Two complementary pipelines plus a validation step:

| Pipeline | Input | Purpose | Calibration |
| --- | --- | --- | --- |
| **Field** | Plant in soil with an AprilTag-marked clipboard in frame | Detect leaves of the *main* plant and compute leaf areas | Template-matched AprilTag (cm/px) |
| **Lab** | Detached leaves on graph paper with AprilTag marker | Ground truth per-leaf area | Template-matched AprilTag (primary) + grid period (cross-check) |
| **Validation** | The two CSV outputs above | Per-plot field-vs-lab comparison + calibration factor | — |

The field pipeline isolates the main plant by **DBSCAN-clustering leaf centroids**
and keeping the cluster nearest the image centre. The lab pipeline **deduplicates**
repeated photos of the same sheet by averaging per-leaf area across the
`A-B-S-H#-1.1, 1.2, 1.3 …` sequence. The validation step computes a per-plot
**calibration factor** = `median(lab top-K) / median(field top-K)` and reports
both raw and calibrated field totals.

## Filename convention

The pipelines parse plot identity from filenames:

- Field: `{TRIAL}-B{block}-S{plot}-{seq}.jpeg` — e.g. `A77-B1-S1-1.1.jpeg`
- Lab:   `{TRIAL}-B{block}-S{plot}-H{leaf_no}-{seq}.jpeg` — e.g. `A77-B1-S1-H1-1.1.jpeg`

Field and lab photos with the same `TRIAL-B#-S#` resolve to the same `plot_key`,
which is how they're matched in validation.

## Install

The project targets the `dl_env` conda environment.

```bash
conda activate dl_env

# Core deps
pip install -r requirements.txt

# SAM3 (gated weights on HuggingFace)
pip install 'git+https://github.com/facebookresearch/sam3.git'

# Runtime deps SAM3 doesn't declare:
pip install 'setuptools<82' pycocotools psutil decord scikit-learn

# Hugging Face auth (one-time)
huggingface-cli login   # paste your token when prompted
```

## Environment variables for runs

```bash
export HF_TOKEN=<your_huggingface_token>          # for SAM3 weight download
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # avoids fragmentation OOM
```

If you'd rather not export them globally, prepend them to each command.

## Run

```bash
conda activate dl_env

# 1. Field photos
python scripts/run_field.py --photos-dir photos_iaf_20260328 --output-dir outputs --downscale 4

# 2. Lab photos
python scripts/run_lab.py   --photos-dir photos_iaf_20260328 --output-dir outputs --downscale 4

# 3. Validation: per-plot comparison + calibration factor
python scripts/validate.py  --output-dir outputs
```

Each step writes per-image CSV + annotated PNG to `outputs/{field,lab}/`,
an aggregated `all_leaves.csv` (and `all_leaves_dedup.csv` for lab), and the
validation step produces `outputs/validation/validation_report.csv` plus the
field-vs-lab scatter.

### Downscale matters

The 4160×2340 source photos are too large for a 6 GB GPU at native resolution.

- `--downscale 4` → 1040×585: fits on a GTX 1060 (6 GB), recommended default
- `--downscale 2` → 2080×1170: requires ≥12 GB VRAM (e.g. T4, RTX 3060+)
- `--downscale 1` → native: needs ≥24 GB VRAM

If you change `--downscale`, **also bump `TemplateMatchingCalibrator.min_side_px`** —
markers shrink with the image. At `ds=8` you'd need `min_side_px≈40`; at `ds=2`,
the default 60 is still fine. Too-small a floor lets the matcher lock onto noise
and returns a wildly wrong cm/px (see project memory `project_aruco_detection.md`).

## Smoke tests (no full directory needed)

To verify the pipeline end-to-end on a tiny subset before committing to a full run:

```bash
# One field + one lab image (default: A77-B1-S1-1.1 and -H1-1.1)
HF_TOKEN=$HF_TOKEN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m scripts.smoke_test --real --downscale 4

# All photos of a single plot (3 field + ~7 lab), including dedup + calibration
HF_TOKEN=$HF_TOKEN PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m scripts.smoke_plot --plot-key A77-B1-S1 --downscale 4
```

`smoke_test.py` can also run *without* a real model — `--stub` (default) injects
synthetic detections to verify the calibration + filtering + output wiring is
sound. Useful when iterating on the code without paying the SAM3 inference cost.

## CLI options

`scripts/run_field.py`:

| Flag | Default | Notes |
|---|---|---|
| `--photos-dir` | `photos_iaf_20260328` | Directory with input JPEGs |
| `--output-dir` | `outputs` | Per-image and summary outputs go here |
| `--downscale` | `2` | Integer image downscale; **use 4 on 6 GB GPUs** |
| `--marker-size-cm` | `9.0` | Physical side length of the AprilTag |
| `--aruco-dict` | `DICT_APRILTAG_16h5` | OpenCV dictionary name |
| `--confidence` | `0.45` | SAM3 confidence threshold |
| `--max-distance-cm` | `30.0` | Centre-distance radius if DBSCAN is disabled |

`scripts/run_lab.py`: same as above, plus `--grid-square-cm` (default `0.5`).

`scripts/validate.py`:

| Flag | Default | Notes |
|---|---|---|
| `--output-dir` | `outputs` | Where to read field/lab CSVs and write validation outputs |
| `--field-csv` | `<output>/field/all_leaves.csv` | Override the input CSV |
| `--lab-csv` | `<output>/lab/all_leaves_dedup.csv` | Override the input CSV |

## Outputs

```
outputs/
  field/
    <image>_leaves.csv           # one row per leaf in main plant cluster
    <image>_leaves.png           # annotated overlay
    <image>_hist.png             # leaf-area histogram
    all_leaves.csv               # concatenation of every image
  lab/
    <image>_leaves.csv           # one row per leaf detection (raw)
    <image>_leaves.png
    all_leaves.csv               # raw concatenation across all lab images
    all_leaves_dedup.csv         # collapsed by (plot_key, leaf_no, rank)
  validation/
    validation_report.csv        # per-plot summary + calibration_factor
    validation_scatter.png       # field vs lab top-K medians
```

Each row in `validation_report.csv`:

- `field_total_area_cm2`, `field_mean_area_cm2` — uncalibrated
- `calibration_factor` — `lab_top_k_median / field_top_k_median`
- `field_total_area_cm2_calibrated` — field × factor, comparable to lab
- `calibration_k` — number of leaves used in the pairing (= min(lab_n, field_n))

## Project layout

```
beanleafmapper/
  config.py              # dataclass defaults — marker, prompts, thresholds, dirs
  io_utils.py            # filename parsing (ImageId), image loading
  model.py               # SAM3 wrapper (Sam3Model.detect)
  calibration/
    aruco.py             # cm/px from cv2.aruco markers
    template.py          # cm/px from template-matched marker (handles printed markers without quiet zone)
    grid.py              # cm/px from graph-paper period (lab cross-check)
  detector/
    base.py              # Detector — masks, bboxes, contours, geometry
    leaves.py            # LeavesDetector — DBSCAN main-plant filter, per-leaf metrics
  pipeline/
    field.py             # field workflow
    lab.py               # lab workflow + dedup_lab_sequences
    validation.py        # per-plot field-vs-lab + calibration factor
  visualization.py       # plot helpers (annotated images, histograms)
scripts/
  run_field.py
  run_lab.py
  validate.py
  smoke_test.py          # one field + one lab image (stub or --real)
  smoke_plot.py          # all photos of a chosen --plot-key
```

## Model configuration

The detector is configurable via `ModelConfig` in `beanleafmapper/config.py` and
via CLI flags on `run_field.py` / `run_lab.py`.

### Default

| Field | Default | Effect |
|---|---|---|
| `backend` | `"sam3_image"` | Use SAM3's image model with text prompts (the only backend shipped today). |
| `checkpoint_path` | `None` | Download the public weights from HuggingFace. |
| `device` | `None` (auto) | `cuda` if available, otherwise `cpu`. |
| `compile` | `False` | No `torch.compile`. |

### CLI overrides

```bash
# Run on Ampere+ GPU with torch.compile (~2x speedup)
python scripts/run_field.py --photos-dir photos_iaf_20260328 --output-dir outputs \
    --downscale 2 --compile

# Use a local fine-tuned checkpoint instead of the public weights
python scripts/run_field.py --photos-dir photos_iaf_20260328 --output-dir outputs \
    --checkpoint-path /path/to/sam3_beans.pt

# Force CPU (slow, but works without a GPU at all)
python scripts/run_field.py --photos-dir photos_iaf_20260328 --output-dir outputs \
    --device cpu --downscale 4
```

The same flags exist on `run_lab.py`.

### Switching to a different model entirely

The `--model-backend` flag and `ModelConfig.backend` field are wired so we can
add alternative text-promptable detectors without changing pipeline code.
Today only `sam3_image` is implemented; passing anything else raises
`NotImplementedError`. The table below lists the candidates we'd consider if
SAM3 isn't a good fit for a given GPU/use case — each would need a new backend
class implementing the same interface (`detect(image, text_prompt) -> Inference`).

| Candidate backend | What it is | When to consider it | Trade-offs |
|---|---|---|---|
| `sam3_image` (default) | SAM3 image model, text-prompted | Recommended starting point — best in-class text-to-mask | Gated weights; ~4 GB VRAM at ds=4; no Flash Attention pre-Ampere |
| `sam3_image_compile` | Same model, `compile=True` | Ampere+ GPU and you want ~2× faster inference | Ampere+ only (sm_80+); first run pays a compile cost |
| `grounding_dino + sam2` | Grounding DINO does text→box, SAM 2 does box→mask | Smaller total model (~250 MB); proven leaf-detection precedent | Two-stage; weaker on densely overlapping leaves than SAM3 |
| `owlv2 + sam2` | OWLv2 for open-vocab boxes, SAM 2 for masks | Open-vocab promotes generalisation to new species/conditions | Same two-stage cost; OWLv2 confidence calibration differs |
| `yolo_world + sam2` | YOLO-World (real-time text-promptable detector) + SAM 2 | Fastest CPU/edge option | Less accurate on the kind of overlapping foliage we see |
| `sam3_video` | SAM3 video predictor (`build_sam3_predictor(version="sam3" | "sam3.1")`) | Time-series of plant photos where you want tracking across days | Currently not needed for still photos; SAM 3.1 adds multiplex tracking |
| Fine-tuned SAM3 | Custom checkpoint trained on labelled bean photos | If you have ground-truth masks and want a meaningful accuracy bump | Need annotated training data + training run; use `--checkpoint-path` to load |

To add one of these, drop a `<Name>Model` class in `beanleafmapper/model.py`
implementing the same `detect(image, text_prompt, confidence_threshold)` signature
that returns a `Sam3Inference`-compatible object (just `.boxes`, `.masks`,
`.scores` tensors), then extend the dispatch in `build_detector()`.

### Note on "bigger SAM3"

There is **no `base / large / huge` size variant of the SAM3 image model** — it
is a single architecture. On a better GPU the levers are: lower `--downscale`,
`--compile`, and bf16/Flash Attention (which auto-activate on Ampere+). A
fine-tuned checkpoint via `--checkpoint-path` is the only path to a *different*
SAM3 image model.

## GPU-tier guidance

| GPU tier | Recommended flags | Notes |
|---|---|---|
| GTX 1060 / 1080 (6 GB, Pascal) | `--downscale 4` | bf16 unsupported → falls back to fp16; no Flash Attention; no `--compile` |
| RTX 2060–2080 / T4 (Turing) | `--downscale 2` | fp16 ok; Flash Attention v1 ok; `--compile` may not help much |
| RTX 30xx / 40xx (Ampere/Ada, 12–24 GB) | `--downscale 2 --compile` | bf16 + Flash Attention 2; `--compile` worth it after first warmup |
| A100 / H100 (40–80 GB) | `--downscale 1 --compile` | Native resolution feasible; Flash Attention 3 |
| CPU only | `--device cpu --downscale 4` | Expect minutes per image, not seconds |

## Known caveats

- **SAM3 weights are gated.** You'll need a HuggingFace account with access granted to the SAM3 model card. First run downloads the weights into the HF cache.
- **GPU memory.** On a GTX 1060 (6 GB), use `--downscale 4`. Lower causes OOM during the mask upsample.
- **AprilTag detection.** The markers in this dataset (`DICT_APRILTAG_16h5` id=0, 9 cm) were printed without the black quiet zone, so standard ArUco/AprilTag detectors fail. The template-matching fallback solves this — see `beanleafmapper/calibration/template.py`.
- **Grid calibration at ds≥4** picked up a harmonic instead of the fundamental period in one test; trust the template matcher. The grid value in the lab CSV is informational.
- **Calibration factor** is per-plot, not global. With more plots the variance across the dataset will tell you whether a single corrective factor is appropriate or whether each plot needs its own.
