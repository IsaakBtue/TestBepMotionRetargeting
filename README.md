# BEP, Humanoid Motion Capture & Retargeting

**Bachelor End Project**  
Pipeline for converting optical motion capture (MoCap) data into robot motion for the Booster T1 humanoid, using SOMA/MoSh++ for body fitting and GMR for retargeting.

**From suit to CSV:** If you are capturing data with the lab OptiTrack system, start with **[handbook.md](handbook.md)**, it walks you from logging in and powering the rig through Motive (calibration files, recording, labeling, export) until you have a CSV ready to copy into `data/` and run the pipeline below.

---

## Table of Contents

1. [Overview](#overview)
2. [OptiTrack capture handbook](handbook.md) (wear suit → exported CSV)
3. [Repository Structure](#repository-structure)
4. [Environment Setup](#environment-setup)
   - [SOMA / MoSh++ Environment (Python 3.7)](#1-soma--mosh-environment-python-37)
   - [Retargeting Environment (Python 3.10)](#2-retargeting-environment-python-310)
5. [Body Models & Checkpoints](#body-models--checkpoints)
6. [Running the Full Pipeline](#running-the-full-pipeline)
7. [Output Files](#output-files)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The full pipeline takes raw CSV marker data from an optical MoCap system and produces robot motion data for the Booster T1:

```
Raw CSV markers
    └── filter_fill_39markers.py       # Fill gaps, extract 39-marker set
        └── run_39marker_soma_mosh.py  # MoSh++ fits SMPL-X body (soma env)
            └── retarget.sh            # GMR retargets SMPL-X to Booster T1 (HumanoidDataGeneration env)
                └── visualize.sh       # MuJoCo viewer
```

---

## Repository Structure

```
BEP/
├── README.md                    # This file
├── handbook.md                  # OptiTrack Motive: from login to exported CSV
│
├── src/
│   ├── pipeline/                # Python pipeline scripts (MoCap → SMPL-X)
│   │   ├── filter_fill_39markers.py
│   │   ├── run_39marker_soma_mosh.py
│   │   ├── csv_to_c3d.py
│   │   ├── visualize_markers.py
│   │   └── visualize_smplx_body.py
│   └── retargeting/             # GMR retargeting module + README
│
├── scripts/                     # Shell scripts to run the pipeline
│   ├── pipeline.sh              # Full end-to-end pipeline
│   ├── retarget.sh              # SMPL-X → robot retargeting
│   ├── retarget_direct.sh       # Direct retarget (skip MoSh++)
│   └── visualize.sh             # Open MuJoCo viewer
│
├── moshpp/                      # MoSh++ source (body solver, actively used)
├── soma/                        # SOMA source (present but not called, labels already known)
├── soma_work/                   # MoSh++ working directory (intermediate outputs, support files)
│
├── data/                        # Input MoCap CSV files
├── output/                      # Pipeline outputs (SMPL-X .pkl, retargeting .pkl)
├── body_models/                 # SMPL / SMPL-X model files (manual download)
├── assets/                      # Robot MJCF models
└── docs/                        # Documentation, workflow notes
```

---

## Environment Setup

This project uses **two separate conda environments** because the MoSh++/SOMA body fitting step requires Python 3.7, while the GMR retargeting step requires Python 3.10.

### 1. MoSh++ Environment (Python 3.7)

Used for: filling marker gaps, fitting SMPL-X with MoSh++.

> **Note on SOMA:** The conda environment is named `soma` because MoSh++ was originally set up inside SOMA's environment. However, the SOMA Python package itself is **not used** in this pipeline. Marker labels are already known from the CSV column names, so SOMA's auto-labeling step is skipped entirely. Only MoSh++ is called (`from moshpp.mosh_head import MoSh, run_moshpp_once`). The `soma/` source folder and `soma_work/` directory exist in the repo but the SOMA code is never executed, `soma_work/` is only used as a working directory structure for MoSh++ intermediate outputs.

#### 1.1 System dependencies

```bash
sudo apt install libtbb-dev libeigen3-dev
```

#### 1.2 Create the environment

```bash
conda create -n soma python=3.7 -y
conda activate soma
```

#### 1.3 Install MoSh++

Clone and install from the `moshpp/` folder (already present in the repo):

```bash
cd moshpp
pip install -r requirements.txt

cd src/moshpp/scan2mesh/mesh_distance
make

cd ../../../..
python setup.py install
cd ..
```

#### 1.4 Verify

```bash
conda activate soma
python -c "import moshpp; print('MoSh++ OK')"
```

---

### 2. Retargeting Environment (Python 3.10)

Used for: SMPL-X → Booster T1 retargeting via GMR, and MuJoCo visualization.

#### 2.1 Create the environment

```bash
conda create -n HumanoidDataGeneration python=3.10 -y
conda activate HumanoidDataGeneration
```

#### 2.2 Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For a different CUDA version, check [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

#### 2.3 Install GVHMR (optional, for video-based pose extraction)

```bash
cd src/retargeting/GVHMR
pip install -r requirements.txt
pip install -e .
cd ../../..
```

#### 2.4 Install GMR (retargeting library)

```bash
cd src/retargeting
pip install -e .
cd ../..
```

#### 2.5 Fix a common C++ stdlib issue

```bash
conda install -c conda-forge libstdcxx-ng -y
```

#### 2.6 Reinstall GMR to apply all fixes

```bash
pip uninstall general-motion-retargeting -y
cd src/retargeting
pip install -e .
cd ../..
```

#### 2.7 Verify

```bash
conda activate HumanoidDataGeneration
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import smplx; print('SMPLX OK')"
python -c "import mujoco; print('MuJoCo OK')"
```

CUDA should print `True`. If it prints `False`, check `nvidia-smi` and your CUDA installation.

---

## Body Models & Checkpoints

You must manually download the SMPL and SMPL-X body models (registration required).

| Model  | Download URL                          |
|--------|---------------------------------------|
| SMPL   | https://smpl.is.tue.mpg.de/          |
| SMPL-X | https://smpl-x.is.tue.mpg.de/        |

After downloading, place files in:

```
body_models/
├── smpl/
│   ├── SMPL_NEUTRAL.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_FEMALE.pkl
└── smplx/
    ├── SMPLX_NEUTRAL.npz
    ├── SMPLX_MALE.npz
    └── SMPLX_FEMALE.npz
```

SMPL-X is **required** for the pipeline. SMPL is only needed for some rendering utilities.

For GVHMR checkpoints (only needed for the video-based route), see `src/retargeting/README.md`.

---

## Running the Full Pipeline

All scripts are run from the repository root (`/home/isaak/BEP`).

### Step 1: Filter and fill marker data

```bash
conda activate soma
python src/pipeline/filter_fill_39markers.py data/your_take.csv
```

This produces `data/your_take_filled39.csv`.

### Step 2: Fit SMPL-X body with MoSh++

```bash
python src/pipeline/run_39marker_soma_mosh.py --csv data/your_take_filled39.csv
```

Output is written to `output/soma_mosh39/`.

### Step 3: Retarget to Booster T1

```bash
conda activate HumanoidDataGeneration
./scripts/retarget.sh
```

Or run the full pipeline in one command:

```bash
./scripts/pipeline.sh --csv data/your_take.csv
```

Optional flags:
- `--skip-smooth`, skip Gaussian smoothing of the SMPL-X fit
- `--compare`, run both raw and smoothed variants side-by-side

### Step 4: Visualize in MuJoCo

```bash
./scripts/visualize.sh
```

---

## Output Files

| Path | Description |
|------|-------------|
| `output/soma_mosh39/` | MoSh++ SMPL-X parameters per frame |
| `output/retargeting/` | Retargeted Booster T1 motion (`.pkl`) |
| `output/retargeting_raw/` | Retargeted without smoothing (with `--compare`) |
| `output/retargeting_smoothed/` | Retargeted with smoothing (with `--compare`) |

The `.pkl` motion files have the following structure:

```python
{
    'rate': 30,        # FPS
    'trans': [...],    # Base position trajectory
    'base_rot': [...], # Base rotation (quaternion)
    'dof_pos': [...],  # Joint angles per frame
}
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `conda activate soma` fails | Make sure `conda init bash` has been run and shell was restarted |
| `make` fails in `scan2mesh/mesh_distance` | Run `sudo apt install libeigen3-dev libtbb-dev` first |
| CUDA not detected (`False`) | Check `nvidia-smi`; reinstall correct PyTorch CUDA build |
| MoSh++ import error | Re-run `python setup.py install` inside `moshpp/` with the `soma` env active |
| GMR import error | Re-run `pip install -e .` inside `src/retargeting/` with `HumanoidDataGeneration` env active |
| Out of memory during retargeting | Use a shorter capture or reduce batch size |

---

## Requirements Summary

| Component | Environment | Python | Key Deps |
|-----------|-------------|--------|----------|
| MoSh++ (SOMA env, SOMA code not used) | `soma` | 3.7 | chumpy, numpy, libtbb, libeigen3 |
| Retargeting (GMR) | `HumanoidDataGeneration` | 3.10 | PyTorch, smplx, mujoco, CUDA 12.1+ |

**Hardware:** NVIDIA GPU with 8 GB+ VRAM recommended. Ubuntu 22.04.
