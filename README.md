# AI‑ENE End‑to‑End (Head & Neck) — Segmentation + ENE Classification

## Overview
This repository provides an end‑to‑end pipeline to:
- Ingest head‑and‑neck DICOM or NRRD/NIfTI, preprocess, and run nnU‑Net v1 segmentation of lymph nodes
- Classify per‑node ENE likelihood and node positivity with AI‑ENE (TensorFlow/Keras)
- Produce a timestamped CSV of predictions and metrics (volumes, optional short/long axes)

Structure (run all commands from the `AI_ENE` folder):
- `nnunet_segmentation`: DICOM/NRRD/NIfTI → preprocessing → nnU‑Net inference
- `ENE_inference`: ENE inference from CT + segmentation
- `run_e2e.py`: wrapper that orchestrates the full flow

## Environment Setup

### Option A: Conda (recommended)
```bash
# From inside AI_ENE/ create environment named ai-ene
conda env create -f environment.yml
conda activate ai-ene

# If PyTorch is not installed (or you want a specific build), install one:
# CPU-only:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# CUDA 11.8 (if you have NVIDIA drivers/CUDA runtime):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install local utils used by AI‑ENE (editable)
pip install -e ENE_inference/data-utils
```

### Option B: pip only
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional: pick a specific PyTorch build if needed (see above)
# Then install local utils
pip install -e ENE_inference/data-utils
```

Notes:
- CPU-only works by default. For GPU, install a CUDA‑compatible TensorFlow build or use Docker.
- nnU‑Net v1 (1.7.0) is used for segmentation; TensorFlow 2.11 for AI‑ENE classification.

## Downloads

- Download required assets from the shared folder: [Google Drive — AI_ENE assets](https://drive.google.com/drive/folders/1XGSU5Jc85SJu-s9vdGAp44LjZafH4tOD?usp=sharing)

- Shared folder structure:
  - AI_ENE/ENE_inference
    - ene_model
      - contains the ENE model file
    - nnunetfile
      - nnUNet_trained_model
      - reg_img_template

- After download, place them into this repository with the following layout:
  - `AI_ENE/ENE_inference/ene_model/0208-1531-1_DualNet.h5`
  
  - `AI_ENE/nnunet_segmentation/nnUNet_trained_model` (folder): should contain
    - `AI_ENE/nnunet_segmentation/nnUNet_trained_model/nnUNet/Task501_ENE/plans.pkl`
    - `AI_ENE/nnunet_segmentation/nnUNet_trained_model/nnUNet/Task501_ENE/fold_0/model_final_checkpoint.model`
    - `AI_ENE/nnunet_segmentation/nnUNet_trained_model/nnUNet/Task501_ENE/fold_0/model_final_checkpoint.model.pkl`

  - `AI_ENE/nnunet_segmentation/reg_img_template/template_image.nrrd`
    - This folder is kept as a placeholder in git; download the template locally (not tracked).




## How to Run — End‑to‑End

### DICOM inputs
```bash
conda activate ai-ene
python run_e2e.py \
  --input-dir /abs/path/to/dicom_root \
  --input-format dicom \
  --output-dir output \
  --gpu cpu
```


 

### Non‑DICOM inputs (NRRD/NIfTI)
```bash
conda activate ai-ene
python run_e2e.py \
  --input-dir /abs/path/to/nifti_or_nrrd \
  --input-format nii.gz \
  --output-dir output \
  --gpu cpu
```
Notes (non‑DICOM):
- Allowed `--input-format`: `nrrd`, `nii`, `nii.gz` (case‑insensitive). If omitted, the value in `nnunet_segmentation/src/config.yaml` is used (defaults to `dicom`).
- You can also set `path.input_data_dir` and `data.input_data_format` in `nnunet_segmentation/src/config.yaml` instead of CLI.

### CLI options
- `--input-dir PATH`: unified input directory. If `--input-format dicom`, this is the DICOM root; otherwise it must contain NRRD/NIfTI. Takes precedence over legacy flags when provided.
- `--input-format {dicom|nrrd|nii|nii.gz}`: input modality. If omitted, uses `data.input_data_format` from `nnunet_segmentation/src/config.yaml` (default `dicom`).
- `--dicom-dir PATH` (legacy): DICOM input directory. Ignored when `--input-dir` is provided.
- `--input-data-dir PATH` (legacy): non‑DICOM input directory (NRRD/NIfTI). Ignored when `--input-dir` is provided.
- `--output-dir PATH`: base output directory for AI‑ENE results (default: `AI_ENE/output`).
- `--gpu {cpu|auto|0}`: device selection (default: `cpu`). Use a CUDA‑enabled setup for GPU.
- `--include-short-axes`: compute short/long axis metrics and include them in the CSV.
- `--limit N`: maximum number of cases to process in the classification step (does not limit nnU‑Net preprocessing/segmentation).
- `--fg-labels {2|any|1,2}`: foreground label(s) in segmentation; `any` treats any nonzero voxel as foreground (default: `2`).

Notes:
- The end-to-end wrapper runs the classification step with verbose logging enabled by default, so you will see detailed progress and a run log saved alongside outputs.
- The runner always uses a fresh, isolated workspace per run under `nnunet_segmentation/runs/run_YYYYMMDD_HHMMSS/` to avoid reusing intermediates across runs.
  - nnU‑Net images (inputs to classifier): `nnunet_segmentation/runs/run_YYYYMMDD_HHMMSS/nnUNet_output/nnUNet_raw_data/Task501_ENE/imagesTs/`
  - nnU‑Net predictions (segmentations): `nnunet_segmentation/runs/run_YYYYMMDD_HHMMSS/nnUNet_output/nnUNet_raw_data/Task501_ENE/predsTs/`



## Run with Docker (single image: GPU by default; CPU optional)

Build the image (from inside `AI_ENE/`):
```bash
docker build -t ai-ene:latest .
```

Run on GPU (default):
```bash
docker run --rm -it --gpus all \
  -v "/abs/path/to/your_dicom_root":/data/dicom \
  -v "$(pwd)":/app \
  -w /app \
  ai-ene:latest \
  --input-dir /data/dicom \
  --input-format dicom \
  --output-dir /app/output \
  --gpu 0
```

Run on CPU (override default):
```bash
docker run --rm -it \
  -v "/abs/path/to/your_dicom_root":/data/dicom \
  -v "$(pwd)":/app \
  -w /app \
  ai-ene:latest \
  --input-dir /data/dicom \
  --input-format dicom \
  --output-dir /app/output \
  --gpu cpu
```

Prerequisites for GPU on host:
- NVIDIA driver installed
- NVIDIA Container Toolkit installed (so Docker supports `--gpus all`)

Quick checks:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
```

Notes (classification on GPU):
- The Docker image includes a GPU-capable TensorFlow (2.13). Classification uses GPU when you run with `--gpus all` and pass `--gpu 0`.
- To force CPU classification, pass `--gpu cpu`.

## Running Components Separately

### nnU‑Net segmentation only
```bash
conda activate ai-ene
cd nnunet_segmentation
python src/run_pipeline.py
```
Outputs:
- Images: `nnUNet_output/nnUNet_raw_data/Task501_ENE/imagesTs/{case_id}_0000.nii.gz`
- Segmentations: `nnUNet_output/nnUNet_raw_data/Task501_ENE/predsTs/{case_id}.nii.gz`

### AI‑ENE classification only(Start from Nifti inputs)
```bash
conda activate ai-ene
python ENE_inference/ene_classification.py \
  --seg-dir nnunet_segmentation/nnUNet_output/nnUNet_raw_data/Task501_ENE/predsTs \
  --img-dir nnunet_segmentation/nnUNet_output/nnUNet_raw_data/Task501_ENE/imagesTs \
  --output-dir output \
  --gpu cpu --verbose --include-short-axes
```

Outputs (classification):
- A timestamped CSV under `AI_ENE/output/prediction_list/predictions_{YYYYMMDD_HHMMSS}/prediction_list_with_volume_short_axis_target_cohort_{YYYYMMDD_HHMMSS}.csv`
- Per-node crops and intermediates under `output_{YYYYMMDD_HHMMSS}/` and `label_croptop_172x172x76_{YYYYMMDD_HHMMSS}/`

## Input requirements for AI‑ENE
- Images: `{case_id}_0000.nii.gz`
- Segmentations: `{case_id}.nii` or `{case_id}.nii.gz`
- Images and segmentations must be co‑registered (same origin, spacing, direction)
- Default LN foreground label is `2`; if your masks use different labels, pass `--fg-labels any` or a list like `--fg-labels 1,2`

