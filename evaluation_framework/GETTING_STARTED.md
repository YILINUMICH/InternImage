# InternImage DCNv3 Evaluation Framework - Complete Setup

## 🎯 Project Mission

Systematically evaluate InternImage's DCNv3 operator for autonomous driving perception to verify claims about:
- Adaptive receptive fields improving distant object detection
- Better safety metrics for pedestrian detection
- Real-time viability (≥30 FPS)
- Cross-dataset generalization

## 📁 Framework Structure

```
InternImage/
├── evaluation_framework/          # ← YOUR EVALUATION CODE
│   ├── README.md                  # Framework overview
│   ├── PROJECT_GUIDE.md           # Detailed experimental guide
│   ├── quick_start.py             # Easy-to-use evaluation launcher
│   ├── requirements.txt           # All dependencies
│   │
│   ├── configs/                   # Experiment configurations
│   │   ├── waymo_experiments.yaml
│   │   ├── nuscenes_experiments.yaml
│   │   └── baseline_comparisons.yaml
│   │
│   ├── scripts/                   # Evaluation scripts
│   │   ├── setup_environment.py   # Environment setup automation
│   │   ├── run_waymo_eval.py      # Waymo evaluation (TODO)
│   │   ├── run_nuscenes_eval.py   # nuScenes evaluation (TODO)
│   │   ├── run_baseline_comparison.py
│   │   ├── measure_fps.py
│   │   └── cross_dataset_test.py
│   │
│   ├── analysis/                  # Analysis tools
│   │   ├── distance_analysis.py   # Distance-based performance
│   │   ├── pedestrian_analysis.py # Safety-critical metrics
│   │   ├── fps_analysis.py        # Runtime performance
│   │   ├── cross_dataset_analysis.py
│   │   └── visualize_results.py
│   │
│   ├── data/                      # Datasets (you'll download these)
│   │   ├── waymo/
│   │   └── nuscenes/
│   │
│   ├── models/                    # Pre-trained models
│   │   └── pretrained/
│   │
│   ├── results/                   # Experiment outputs
│   │   ├── waymo/
│   │   ├── nuscenes/
│   │   ├── baselines/
│   │   └── cross_dataset/
│   │
│   └── benchmarks/                # Performance benchmarks
│
├── detection/                     # Original InternImage detection code
├── autonomous_driving/            # Original InternImage AV code
└── ...
```

## 🚀 Quick Start (5 Steps)

### Step 1: Setup Environment

```powershell
# Navigate to evaluation framework
cd evaluation_framework

# Run automated setup
python scripts/setup_environment.py

# OR manually install dependencies
pip install -r requirements.txt

# Install mmcv and mmdetection
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install mmdet==2.28.1

# Compile DCNv3 operators
cd ../detection/ops_dcnv3
sh make.sh
python test.py  # Should show "All checks passed"
cd ../../evaluation_framework
```

### Step 2: Download Datasets

**Waymo Open Dataset:**
```powershell
# Download from: https://waymo.com/open/
# Place in: evaluation_framework/data/waymo/
# Structure:
#   data/waymo/
#   ├── train/
#   │   ├── segment-*.tfrecord
#   └── val/
#       ├── segment-*.tfrecord
```

**nuScenes:**
```powershell
# Download from: https://www.nuscenes.org/
# Place in: evaluation_framework/data/nuscenes/
# Structure:
#   data/nuscenes/
#   ├── samples/
#   ├── sweeps/
#   └── v1.0-trainval/
```

### Step 3: Download Pre-trained Models

```powershell
# Download InternImage models from HuggingFace
# https://huggingface.co/OpenGVLab/InternImage

# Place in: evaluation_framework/models/pretrained/

# Required models:
# - mask_rcnn_internimage_t_fpn_1x_coco.pth
# - mask_rcnn_internimage_s_fpn_1x_coco.pth
# - mask_rcnn_internimage_b_fpn_1x_coco.pth

# Example download (using wget):
cd models/pretrained
wget https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_1x_coco.pth
wget https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_s_fpn_1x_coco.pth
cd ../..
```

### Step 4: Run Evaluations

**Option A: Quick Start (Recommended)**
```powershell
# Check environment
python quick_start.py --check-only

# Run Waymo evaluation
python quick_start.py --dataset waymo --model internimage_s

# Run nuScenes evaluation
python quick_start.py --dataset nuscenes --model internimage_s

# Run all models
python quick_start.py --dataset waymo --model all
```

**Option B: Manual Execution**
```powershell
# Waymo evaluation
python scripts/run_waymo_eval.py `
    --config configs/waymo_experiments.yaml `
    --output results/waymo

# nuScenes evaluation
python scripts/run_nuscenes_eval.py `
    --config configs/nuscenes_experiments.yaml `
    --output results/nuscenes

# Baseline comparison
python scripts/run_baseline_comparison.py `
    --models internimage_s resnet50 `
    --dataset waymo `
    --output results/baselines
```

### Step 5: Analyze Results

```powershell
# Distance-based analysis
python analysis/distance_analysis.py `
    --results results/waymo `
    --output results/waymo/analysis

# Pedestrian safety analysis
python analysis/pedestrian_analysis.py `
    --results results/waymo `
    --output results/waymo/analysis

# FPS benchmarking
python analysis/fps_analysis.py `
    --results benchmarks/ `
    --output results/analysis

# Generate visualizations
python analysis/visualize_results.py `
    --input results/ `
    --output reports/
```

## 📊 Key Experiments

### Experiment 1: Waymo Detection
**Goal:** Evaluate DCNv3 on Waymo Open Dataset  
**Config:** `configs/waymo_experiments.yaml`  
**Focus:** Distant object detection (>50m), pedestrian recall

### Experiment 2: nuScenes Detection
**Goal:** Evaluate DCNv3 on nuScenes dataset  
**Config:** `configs/nuscenes_experiments.yaml`  
**Focus:** Multi-class detection, 360° camera coverage

### Experiment 3: Baseline Comparisons
**Goal:** Compare DCNv3 vs. standard CNN backbones  
**Baselines:** ResNet-50, Swin Transformer, ConvNeXt  
**Metrics:** mAP, FPS, parameter count

### Experiment 4: Cross-Dataset Generalization
**Goal:** Test domain transfer (Waymo ↔ nuScenes)  
**Analysis:** Performance degradation, camera sensitivity

### Experiment 5: FPS Benchmarking
**Goal:** Measure real-time inference performance  
**Target:** ≥30 FPS on RTX 3090  
**Tests:** Various batch sizes, resolutions

## 📈 Key Metrics

### Detection Performance
- **mAP**: Overall detection quality
- **AP50, AP75**: Different IoU thresholds
- **AP by distance**: Binned by distance ranges
- **AP by size**: Small, medium, large objects
- **Per-class AP**: Especially pedestrian, cyclist

### Safety Metrics (Critical)
- **Pedestrian Recall**: Must be >90%
- **Miss Rate**: Target <10% at all distances
- **False Positive Rate**: Balance with recall
- **Critical Distance Performance**: <10m detection

### Runtime Performance
- **FPS**: Target ≥30 for real-time
- **Latency**: Per-frame inference time
- **GPU Memory**: Fit on typical GPUs
- **Throughput**: Images processed per second

### Generalization
- **Cross-dataset mAP**: Performance on unseen data
- **Domain Gap**: Performance drop across datasets
- **Camera Model Sensitivity**: Robustness to camera changes

## 🔬 Analysis Tools

### 1. `distance_analysis.py`
Analyzes detection performance across distance ranges.

**Outputs:**
- AP by distance bins
- Performance comparison heatmaps
- Detection distribution histograms
- Summary report

**Usage:**
```powershell
python analysis/distance_analysis.py --results results/waymo
```

### 2. `pedestrian_analysis.py`
Safety-critical pedestrian detection analysis.

**Outputs:**
- Miss rate by distance
- Recall comparison charts
- Precision-recall curves
- Safety assessment report

**Usage:**
```powershell
python analysis/pedestrian_analysis.py --results results/waymo
```

### 3. `fps_analysis.py`
Real-time performance benchmarking.

**Outputs:**
- FPS comparison charts
- Latency measurements
- Memory usage analysis
- Batch size scaling plots

**Usage:**
```powershell
python analysis/fps_analysis.py --results benchmarks/
```

## 🎓 Expected Learning Outcomes

After completing this evaluation, you will be able to:

1. **Critically assess research claims** through systematic experimentation
2. **Design reproducible evaluation pipelines** for computer vision models
3. **Analyze performance across multiple dimensions** (accuracy, speed, safety)
4. **Identify practical limitations** of academic models in real-world scenarios
5. **Communicate technical findings** through reports and visualizations

## ⚠️ Important Notes

### Implementation Status
- ✅ Framework structure created
- ✅ Configuration files ready
- ✅ Analysis tools implemented
- ⚠️ Dataset-specific evaluation scripts need implementation
- ⚠️ Model loading code needs integration with InternImage

### Next Development Steps
1. Implement `run_waymo_eval.py` using InternImage's detection code
2. Implement `run_nuscenes_eval.py` using autonomous_driving code
3. Adapt BEVFormer evaluation code for nuScenes
4. Integrate Waymo evaluation from existing configs
5. Test end-to-end pipeline with sample data

### Where to Find Existing Code
- **Waymo configs:** `autonomous_driving/Online-HD-Map-Construction/src/configs/_base_/datasets/waymoD5-*.py`
- **nuScenes configs:** `autonomous_driving/occupancy_prediction/projects/configs/`
- **Detection code:** `detection/` directory
- **BEVFormer models:** `autonomous_driving/occupancy_prediction/`

## 🤝 Collaboration Points with Copilot

Copilot can help with:

1. **Implementing dataset-specific evaluation scripts**
   - Integrating Waymo data loaders
   - Adapting nuScenes evaluation code
   - Model inference pipelines

2. **Debugging issues**
   - CUDA/compilation problems
   - Dataset loading errors
   - Metric calculation bugs

3. **Creating additional analysis tools**
   - Custom visualizations
   - Statistical tests
   - Ablation studies

4. **Optimizing performance**
   - Batch processing
   - Multi-GPU support
   - TensorRT conversion

5. **Generating reports**
   - LaTeX tables
   - Publication-quality figures
   - Technical documentation

## 📚 Documentation

- **Framework README:** `evaluation_framework/README.md`
- **Project Guide:** `evaluation_framework/PROJECT_GUIDE.md`
- **InternImage Docs:** `README.md` (root)
- **Detection Docs:** `detection/README.md`
- **Autonomous Driving:** `autonomous_driving/*/README.md`

## 🐛 Troubleshooting

See `PROJECT_GUIDE.md` section "Troubleshooting" for common issues and solutions.

## 📞 Support

For help:
1. Check documentation in each directory
2. Review InternImage GitHub issues
3. Consult the original CVPR paper
4. Ask Copilot for specific implementation help

## 🎉 You're Ready!

Run this to verify your setup:
```powershell
python quick_start.py --check-only --show-commands
```

Then start with a simple evaluation:
```powershell
python quick_start.py --dataset waymo --model internimage_s
```

Good luck with your evaluation! 🚀

## 🖥️ GPU Compatibility (RTX 5080 / Blackwell sm_120)

Your current GPU (RTX 5080, Blackwell architecture – CUDA compute capability `sm_120`) is ahead of the officially supported architectures in the PyTorch builds we tested (stable 2.5.1 + cu121 and nightly 2.7.0.dev + cu124). As of November 2025:

- Supported architectures in public wheels: `sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90`
- Missing architecture: `sm_120` (Blackwell). PyTorch emits a warning and compiled CUDA extensions fail with: `RuntimeError: CUDA error: no kernel image is available for execution on the device`.
- DCNv3 kernels (and other custom ops) cannot execute because no binary code or PTX path can JIT to the new architecture yet.

### What This Means
You can finish framework setup (datasets, configs, analysis tools) but any evaluation requiring DCNv3 GPU kernels or mmcv CUDA ops will not run on the RTX 5080 until PyTorch adds native Blackwell support.

### Verification Commands
Use the provided script (added in `scripts/check_gpu_support.py`) or run manually:
```powershell
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0)); print('Is Available:', torch.cuda.is_available())"
```
Check compiled arch list (may not include sm_120):
```powershell
python -c "import torch; print(torch.cuda.get_arch_list())"
```
If `sm_120` (or `compute_120`) is absent, the wheel cannot natively target Blackwell.

### Workaround Options
1. Use a supported GPU (e.g., RTX 3090/4090, A100, H100) for experiments.
2. Wait for an official PyTorch release (likely ≥ 2.6) including Blackwell.
3. Periodically test new nightlies: 
    ```powershell
    pip uninstall torch torchvision -y
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu12x
    ```
    (Replace `cu12x` with the newest CUDA minor once Blackwell support lands.)
4. Try a source build (advanced; may still fail until upstream patches merge):
    ```powershell
    git clone --recursive https://github.com/pytorch/pytorch.git
    cd pytorch
    setx TORCH_CUDA_ARCH_LIST "9.0+PTX"   # Temporary; sm_120 unsupported
    pip install -r requirements.txt
    python setup.py develop
    ```
    If build errors mention unknown arch `12.0` or missing code generation, upstream support is still pending.

### Why PTX Didn’t Help
Although compiling with `9.0+PTX` allows PTX fallback for future GPUs, the low-level kernels (DCNv3) still rely on PyTorch’s dispatcher and generated code paths that expect recognized architectures. Without explicit `sm_120` support, PTX JIT cannot bridge all gaps for these custom ops.

### Recommended Interim Plan
- Proceed with dataset preparation and analysis tooling development.
- Mock evaluation outputs using saved results from a supported GPU environment (if available) to exercise analysis scripts.
- Track PyTorch release notes / GitHub issues related to “Blackwell” or “sm_120” enabling.

### Quick Mock Strategy (Optional)
You can simulate detection outputs by placing JSON result files under `results/waymo/` or `results/nuscenes/` that match expected structure, then run analysis scripts to validate logic without GPU inference.

---
