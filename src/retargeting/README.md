# Techunited Data Generation

**Video to Robot Motion Pipeline for Booster T1**

Transform any video of human movement into robot motion data for the Booster T1 humanoid.

---

## 🎬 Demo Videos

See the pipeline in action - from human video to robot motion:

**Demo 1:** [GVHMR Pose Extraction + Robot Retargeting](assets/booster_t1/booster_t1_hmr4d_results.mp4)  
*GVHMR extracts human pose, GMR retargets to Booster T1*

**Demo 2:** [Complex Motion - Full Pipeline](assets/booster_t1/Intercept4_3_incam_global_horiz.mp4)  
*Human performing dynamic movement → Robot motion visualization in MuJoCo*

---

## Quick Start (4 Commands)

```bash
# 1. Clone the repository
git clone https://github.com/IsaakBtue/DatagenTechUnited.git
cd DatagenTechUnited

# 2. Setup environment
./setup_environment.sh

# 3. Download SMPL and SMPLX body models manually
# You need to sign up and download SMPL and SMPLX from their official websites:
#   SMPL:  https://smpl.is.tue.mpg.de/
#   SMPLX: https://smpl-x.is.tue.mpg.de/
# Place the files in:
#   GVHMR/inputs/checkpoints/body_models/smpl/SMPL_{GENDER}.pkl
#   GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_{GENDER}.npz

# 4. Download checkpoints and sample video
./download_checkpoints.sh

# 5. Verify installation
./verify_installation.sh

# 6. Run pipeline with sample video
./run_pipeline.sh --video data/intercept1.mp4
```


---

## What This Does

**Pipeline:** `Video → GVHMR (pose extraction) → GMR (retargeting) → MuJoCo (visualization)`

Takes a video of a human performing an action and generates:
- Robot motion data (.pkl files)
- Vsualization video (.mp4)
- Frame-by-frame joint angles and trajectories

---

## Requirements

- **OS:** Ubuntu 22.04 (Linux)
- **Python:** 3.10
- **GPU:** NVIDIA GPU with 8GB+ VRAM (CUDA 12.1+)
- **Conda:** Miniconda or Anaconda
- **Disk Space:** ~15GB (models + environment)

---

## Detailed Setup Guide

### Step 1: Setup Environment

This creates the conda environment and installs all dependencies:

```bash
./setup_environment.sh
```

The script will:
- Create a conda environment named `HumanoidDataGeneration` with Python 3.10
- Install PyTorch with CUDA support
- Install GVHMR and GMR libraries
- Install all required dependencies
- Takes ~10-15 minutes

<details>
<summary><b>Manual Environment Setup (Click to expand)</b></summary>

If the automatic setup script doesn't work, you can set up the environment manually:

#### 1. Create Conda Environment

```bash
conda create -n HumanoidDataGeneration python=3.10 -y
conda activate HumanoidDataGeneration
```

#### 2. Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Or visit [PyTorch website](https://pytorch.org/get-started/locally/) for your specific CUDA version.

#### 3. Install GVHMR

```bash
cd GVHMR
pip install -r requirements.txt
pip install -e .
cd ..
```

#### 4. Install GMR

```bash
pip install -e .
```

#### 5. Install Additional Dependencies

```bash
conda install -c conda-forge libstdcxx-ng -y
```

#### 6. Reinstall GMR (Important!)

After everything is installed, reinstall GMR to ensure all fixes are applied:

```bash
pip uninstall general-motion-retargeting -y
pip install -e .
```

#### 7. Verify Installation

```bash
conda activate HumanoidDataGeneration
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import smplx; print('SMPLX installed')"
python -c "import mujoco; print('MuJoCo installed')"
```

All imports should succeed and CUDA should be available (True).

</details>

### Step 2: Download Models

There are two parts to the model setup:

1. **SMPL and SMPLX body models (manual download, required)**
2. **Checkpoints and sample video (scripted download)**

#### 2.1 SMPL and SMPLX body models (manual)

You must sign up and download the body models from the official sites:

- SMPL: `https://smpl.is.tue.mpg.de/`
- SMPLX: `https://smpl-x.is.tue.mpg.de/`

After downloading, place the files in the following structure:

```
GVHMR/inputs/checkpoints/body_models/
├── smpl/
│   └── SMPL_{GENDER}.pkl    # Optional, used for rendering and evaluation
└── smplx/
    └── SMPLX_{GENDER}.npz   # Required, used for SMPLX motion and retargeting
```

Typical filenames are:

- `GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl`
- `GVHMR/inputs/checkpoints/body_models/smpl/SMPL_MALE.pkl`
- `GVHMR/inputs/checkpoints/body_models/smpl/SMPL_FEMALE.pkl`
- `GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`
- `GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_MALE.npz`
- `GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_FEMALE.npz`

SMPLX is required for the pipeline to run. SMPL is only needed for some rendering and evaluation utilities.

#### 2.2 Checkpoints and sample video (script)

Use the provided script to download the remaining checkpoints and the sample video:

```bash
./download_checkpoints.sh
```

This script downloads:
- **Sample video** (intercept1.mp4) – about 50MB  
- **GVHMR checkpoints and detector models** – about 10GB

The script will:
- Install `gdown` if needed
- Download from Google Drive
- Organize files automatically
- Verify all downloads

<details>
<summary><b>Manual Installation (Click to expand)</b></summary>

If the automatic download script doesn't work, you can download files manually:

#### 1. Body Models & Sample Video

Visit: https://drive.google.com/drive/folders/1J6lsvquyDFxZjjeSXo-Q57d82mKCVkn0

Download and place:
- **body_models folder** → Extract to `GVHMR/inputs/checkpoints/`
  - Should contain `smpl/` and `smplx/` subfolders
  - `smpl/` should have: `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl`, `SMPL_FEMALE.pkl`
  - `smplx/` should have: `SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz`
- **intercept1.mp4** → Place in `data/` folder

#### 2. GVHMR Checkpoints

Visit: https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD

Navigate into each subfolder and download:
- From **gvhmr/** folder → `gvhmr_siga24_release.ckpt` → place in `GVHMR/inputs/checkpoints/gvhmr/`
- From **hmr2/** folder → `epoch=10-step=25000.ckpt` → place in `GVHMR/inputs/checkpoints/hmr2/`
- From **vitpose/** folder → `vitpose-h-multi-coco.pth` → place in `GVHMR/inputs/checkpoints/vitpose/`
- From **yolo/** folder → `yolov8x.pt` → place in `GVHMR/inputs/checkpoints/yolo/`
- From **dpvo/** folder → `dpvo.pth` → place in `GVHMR/inputs/checkpoints/dpvo/` (optional)

#### 3. Final Directory Structure

After manual download, your structure should look like:

```
GVHMR/inputs/checkpoints/
├── body_models/
│   ├── smpl/
│   │   ├── SMPL_NEUTRAL.pkl
│   │   ├── SMPL_MALE.pkl
│   │   └── SMPL_FEMALE.pkl
│   └── smplx/
│       ├── SMPLX_NEUTRAL.npz
│       ├── SMPLX_MALE.npz
│       └── SMPLX_FEMALE.npz
├── gvhmr/
│   └── gvhmr_siga24_release.ckpt
├── hmr2/
│   └── epoch=10-step=25000.ckpt
├── vitpose/
│   └── vitpose-h-multi-coco.pth
└── yolo/
    └── yolov8x.pt

data/
└── intercept1.mp4
```

After manual download, run `./verify_installation.sh` to check everything is in place.

</details>

### Step 3: Verify Installation

Check that everything is installed correctly:

```bash
./verify_installation.sh
```

This checks:
- ✓ Conda environment exists
- ✓ Python packages installed
- ✓ Body models present
- ✓ Checkpoints downloaded
- ✓ Sample video available

If verification fails, the script will tell you exactly what's missing.

### Step 4: Run the Pipeline

Process your first video (the included sample):

```bash
./run_pipeline.sh --video data/intercept1.mp4
```

**What happens:**
1. GVHMR extracts human pose from video
2. GMR retargets motion to Booster T1
3. MuJoCo generates visualization
4. Output saved to `outputs/` and `videos/`

**Processing time:** 2-5 minutes for a short video (depends on GPU)

---

## Using Your Own Videos

Once setup is complete, process any video:

```bash
./run_pipeline.sh --video /path/to/your/video.mp4
```

**Video Requirements:**
- Single person clearly visible
- Static camera (or use `--no-skip-vo` for moving camera)
- Good lighting
- .mp4 format

**Example with moving camera:**
```bash
./run_pipeline.sh --video /path/to/video.mp4 --no-skip-vo
```

---

## Output Files

After running the pipeline, you'll find:

**Motion Data:**
- `outputs/<video_name>/<video_name>_t1.pkl` - Robot motion data

**Visualization:**
- `videos/<video_name>_visualization.mp4` - MuJoCo visualization
- `GVHMR/outputs/demo/<video_name>/` - GVHMR intermediate outputs

**Motion data format (.pkl):**
```python
{
    'rate': 30,  # FPS
    'trans': [...],  # Base position trajectory
    'base_rot': [...],  # Base rotation (quaternion)
    'dof_pos': [...],  # Joint angles per frame
}
```

---

## Advanced Options

```bash
./run_pipeline.sh --video <path> [options]

Options:
  --video PATH          Input video file (required)
  --robot NAME          Robot model (default: booster_t1)
  --output-dir DIR      Output directory (default: outputs)
  --no-skip-vo          Use visual odometry for moving camera
  --no-video            Skip visualization video generation
  --help                Show all options
```

**Examples:**

```bash
# Moving camera
./run_pipeline.sh --video video.mp4 --no-skip-vo

# Custom output directory
./run_pipeline.sh --video video.mp4 --output-dir my_outputs

# Skip video generation (faster)
./run_pipeline.sh --video video.mp4 --no-video
```

---

## Troubleshooting

### Installation Issues

**Problem:** `setup_environment.sh` fails  
**Solution:** Check conda is installed: `conda --version`

**Problem:** GPU not detected  
**Solution:** Verify CUDA: `nvidia-smi`

**Problem:** Download fails  
**Solution:** Run `./download_checkpoints.sh` again - it skips existing files

### Runtime Issues

**Problem:** `huggingface-hub` version error  
**Solution:** Run `./fix_dependencies.sh`

**Problem:** Pipeline fails on first run  
**Solution:** Run `./verify_installation.sh` to check what's missing

**Problem:** Out of memory  
**Solution:** Use a shorter video or smaller resolution

### Getting Help

1. Check `./verify_installation.sh` output
2. Read error messages carefully - they usually tell you what's wrong
3. Make sure all models downloaded successfully
4. Try the sample video first: `./run_pipeline.sh --video data/intercept1.mp4`

---

## Project Structure

```
DatagenTechUnited/
├── README.md                    # This file
├── CHANGELOG.md                 # Version history
│
├── setup_environment.sh         # 1. Setup conda environment
├── download_checkpoints.sh      # 2. Download models
├── verify_installation.sh       # 3. Verify setup
├── run_pipeline.sh              # 4. Process videos
├── fix_dependencies.sh          # Fix version conflicts
│
├── GVHMR/                       # GVHMR source code
│   ├── inputs/checkpoints/      # All models stored here
│   │   ├── body_models/         # SMPL/SMPL-X
│   │   ├── gvhmr/               # GVHMR checkpoint
│   │   ├── hmr2/                # HMR2 checkpoint
│   │   ├── vitpose/             # VitPose checkpoint
│   │   └── yolo/                # YOLO checkpoint
│   └── outputs/                 # GVHMR outputs
│
├── general_motion_retargeting/  # GMR library (with fixes)
├── assets/booster_t1/           # Robot models
├── scripts/                     # Utility scripts
│
├── data/                        # Input videos go here
│   └── intercept1.mp4           # Sample video
│
├── outputs/                     # Motion data (.pkl)
└── videos/                      # Visualization videos (.mp4)
```

---

## Credits

This package integrates:
- **GMR** - https://github.com/YanjieZe/GMR (MIT License)
- **GVHMR** - https://github.com/zju3dv/GVHMR
- **MuJoCo** - https://mujoco.org/
- **Booster T1** - https://www.boosterobotics.com/

---

