# Quick Manual Setup Guide for CUDA 12.8

Your system has **CUDA 12.8** installed. Here's the corrected setup process:

## Step 1: Install PyTorch (CUDA 12.x compatible)

```powershell
# Install PyTorch with CUDA 12.1 support (works with CUDA 12.8)
pip install torch==2.9.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

## Step 2: Install Core Dependencies

```powershell
# Install basic dependencies
pip install numpy==1.26.4 opencv-python scipy pyyaml termcolor

# Install mmcv and mmdetection
pip install -U openmim
mim install mmcv-full==1.5.0
pip install mmdet==2.28.1
mim install mmsegmentation==0.27.0

# Install additional packages
pip install timm==0.6.11 pydantic==1.10.13 yapf==0.40.1
pip install matplotlib seaborn pandas tqdm tensorboard
```

## Step 3: Install Dataset Tools

```powershell
# For nuScenes
pip install nuscenes-devkit

# For Waymo (TensorFlow-based)
pip install waymo-open-dataset-tf-2-11-0

# COCO tools
pip install pycocotools
```

## Step 4: Compile DCNv3 Operators

```powershell
# Navigate to DCNv3 directory
cd ..\detection\ops_dcnv3

# On Windows, you may need to use the appropriate shell
# If using PowerShell, run:
bash make.sh

# Test compilation
python test.py

# Return to evaluation framework
cd ..\..\evaluation_framework\scripts
```

## Step 5: Verify Installation

```powershell
# Test PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

Expected output:
```
PyTorch: 2.9.1
CUDA Available: True
CUDA Version: 12.1
```

## Troubleshooting

### If DCNv3 compilation fails:

**Option 1: Use pre-compiled wheels**
```powershell
# Download from: https://github.com/OpenGVLab/InternImage/releases/tag/whl_files
# Install the appropriate .whl file for your Python version
pip install DCNv3-1.0-cp312-cp312-win_amd64.whl
```

**Option 2: Check CUDA toolkit**
```powershell
# Verify nvcc is accessible
nvcc --version

# Make sure Visual Studio C++ compiler is installed
# Download from: https://visualstudio.microsoft.com/downloads/
```

### If mmcv-full installation fails:

```powershell
# Try installing from source or use compatible version
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.9/index.html
```

## Alternative: Run Updated Setup Script

I've updated the setup script to auto-detect CUDA 12.8. Try running it again:

```powershell
# From evaluation_framework/scripts/
python setup_environment.py --skip-pytorch=false
```

Or skip PyTorch installation and do it manually:

```powershell
python setup_environment.py --skip-pytorch
```

## Quick Install (All in One)

Run these commands in sequence:

```powershell
# 1. PyTorch
pip install torch==2.9.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Core ML libraries
pip install -U openmim
mim install mmcv-full==1.5.0
pip install mmdet==2.28.1

# 3. Additional dependencies
pip install numpy==1.26.4 opencv-python scipy pyyaml termcolor timm==0.6.11 matplotlib seaborn pandas tqdm

# 4. Dataset tools
pip install nuscenes-devkit pycocotools

# 5. Verify
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

## Next Steps

After successful installation:

1. **Test basic functionality:**
   ```powershell
   cd ..\..
   python evaluation_framework\quick_start.py --check-only
   ```

2. **Compile DCNv3** (if not done):
   ```powershell
   cd detection\ops_dcnv3
   bash make.sh
   python test.py
   ```

3. **Download pre-trained models** (see GETTING_STARTED.md Step 3)

4. **Download datasets** (see GETTING_STARTED.md Step 2)
