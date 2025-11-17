# InternImage DCNv3 Evaluation Framework for Autonomous Driving

**Project Goal**: Systematically evaluate InternImage's DCNv3 operator claims on real-world self-driving perception tasks using existing code from Waymo and nuScenes datasets.

## Research Questions

1. **Performance**: Does DCNv3 provide adaptive receptive fields that improve detection of small, distant objects compared to regular CNNs?
2. **Safety Focus**: Does DCNv3 improve detection of faraway pedestrians (primary safety metric)?
3. **Generalization**: How well does the model generalize across different camera models and environments (Waymo vs. nuScenes)?
4. **Real-time Viability**: Can DCNv3 achieve ≥30 FPS on typical GPU hardware for practical deployment?
5. **Trade-offs**: What are the computational costs vs. accuracy benefits?

## Directory Structure

```
evaluation_framework/
├── README.md                    # This file
├── configs/                     # Experiment configurations
│   ├── waymo_experiments.yaml
│   ├── nuscenes_experiments.yaml
│   └── baseline_comparisons.yaml
├── scripts/                     # Automated evaluation scripts
│   ├── setup_environment.py
│   ├── run_waymo_eval.py
│   ├── run_nuscenes_eval.py
│   ├── run_baseline_comparison.py
│   ├── measure_fps.py
│   └── cross_dataset_test.py
├── analysis/                    # Analysis and visualization tools
│   ├── distance_analysis.py     # Analyze detection by object distance
│   ├── pedestrian_analysis.py   # Focus on pedestrian detection metrics
│   ├── fps_analysis.py          # Runtime performance analysis
│   ├── cross_dataset_analysis.py # Generalization analysis
│   └── visualize_results.py     # Generate plots and figures
├── results/                     # Experiment outputs
│   ├── waymo/
│   ├── nuscenes/
│   ├── baselines/
│   └── cross_dataset/
└── benchmarks/                  # Performance benchmarks
    ├── detection_metrics.json
    ├── fps_measurements.json
    └── comparison_tables.json
```

## Quick Start

### 1. Environment Setup

```bash
# Navigate to evaluation framework
cd InternImage/evaluation_framework

# Run setup script (installs dependencies, prepares datasets)
python scripts/setup_environment.py
```

### 2. Run Evaluations

```bash
# Run Waymo evaluation
python scripts/run_waymo_eval.py --config configs/waymo_experiments.yaml

# Run nuScenes evaluation
python scripts/run_nuscenes_eval.py --config configs/nuscenes_experiments.yaml

# Run baseline comparisons
python scripts/run_baseline_comparison.py --config configs/baseline_comparisons.yaml
```

### 3. Performance Analysis

```bash
# Analyze detection by distance
python analysis/distance_analysis.py --results results/

# Pedestrian-specific analysis
python analysis/pedestrian_analysis.py --results results/

# Runtime performance
python analysis/fps_analysis.py --results results/

# Cross-dataset generalization
python analysis/cross_dataset_analysis.py --results results/
```

### 4. Generate Reports

```bash
# Generate comprehensive visualization report
python analysis/visualize_results.py --output reports/
```

## Experiments

### Experiment 1: Waymo Detection Evaluation
- **Goal**: Evaluate InternImage on Waymo Open Dataset
- **Metrics**: mAP, AP by distance bins, pedestrian recall
- **Focus**: Distant object detection (>50m)

### Experiment 2: nuScenes Detection Evaluation
- **Goal**: Evaluate InternImage on nuScenes dataset
- **Metrics**: NDS, mAP, per-class AP, distance-based metrics
- **Focus**: Multi-class detection, especially pedestrians

### Experiment 3: Baseline Comparisons
- **Goal**: Compare DCNv3 vs. standard CNN backbones
- **Baselines**: ResNet-50, Swin Transformer, ConvNeXt
- **Metrics**: mAP difference, FPS comparison, parameter count

### Experiment 4: Cross-Dataset Generalization
- **Goal**: Test model trained on one dataset, evaluated on another
- **Tests**: 
  - Waymo → nuScenes
  - nuScenes → Waymo
- **Metrics**: Performance degradation, domain adaptation needs

### Experiment 5: FPS Benchmarking
- **Goal**: Measure real-time inference performance
- **Hardware**: RTX 3090, RTX 4090, A100
- **Batch sizes**: 1, 4, 8
- **Target**: ≥30 FPS for real-time deployment

## Key Metrics

### Detection Performance
- **mAP**: Mean Average Precision (overall detection quality)
- **AP50, AP75**: AP at IoU thresholds 0.5 and 0.75
- **AP_dist**: AP binned by distance (0-30m, 30-50m, >50m)
- **Pedestrian Recall**: Critical safety metric

### Runtime Performance
- **FPS**: Frames per second
- **Latency**: Inference time per frame (ms)
- **Memory**: GPU memory usage (GB)
- **FLOPs**: Computational cost

### Generalization
- **Cross-dataset mAP**: Performance on unseen dataset
- **Domain gap**: Performance drop across datasets
- **Camera sensitivity**: Variance across different camera models

## Analysis Tools

### 1. Distance-based Analysis (`distance_analysis.py`)
- Bin detections by distance ranges
- Compare DCNv3 vs. baseline performance at different distances
- Visualize improvement on distant objects

### 2. Pedestrian Analysis (`pedestrian_analysis.py`)
- Focus on pedestrian class
- Safety-critical metrics (miss rate, false positives)
- Distance-specific pedestrian detection

### 3. FPS Analysis (`fps_analysis.py`)
- Measure inference speed across different GPUs
- Profile computational bottlenecks
- Real-time feasibility assessment

### 4. Cross-dataset Analysis (`cross_dataset_analysis.py`)
- Domain shift quantification
- Camera model sensitivity
- Generalization capabilities

## Expected Outcomes

### If DCNv3 Claims are Valid:
1. **Better distant object detection**: Higher AP for objects >50m
2. **Improved pedestrian recall**: Especially at longer distances
3. **Reasonable runtime**: Achievable 30+ FPS on modern GPUs
4. **Good generalization**: Moderate performance across datasets

### If Claims Need Verification:
1. **Quantify actual improvements**: Marginal vs. significant
2. **Identify limitations**: Where does DCNv3 fail?
3. **Trade-off analysis**: Accuracy vs. speed vs. complexity
4. **Practical recommendations**: When to use DCNv3

## Data Requirements

### Waymo Open Dataset
- Detection split (train/val/test)
- Camera images + 3D annotations
- Download from: https://waymo.com/open/

### nuScenes
- Detection task data
- Multi-camera setup
- Download from: https://www.nuscenes.org/

### Pre-trained Models
- InternImage-T/S/B/L (from HuggingFace)
- Baseline models (ResNet, Swin, etc.)

## Timeline

1. **Week 1**: Environment setup, data preparation
2. **Week 2**: Run Waymo and nuScenes evaluations
3. **Week 3**: Baseline comparisons, FPS benchmarking
4. **Week 4**: Cross-dataset tests, analysis
5. **Week 5**: Report generation, visualization

## Citation

If you use this evaluation framework, please cite:

```bibtex
@article{wang2022internimage,
  title={InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions},
  author={Wang, Wenhai and Dai, Jifeng and Chen, Zhe and Huang, Zhenhang and Li, Zhiqi and Zhu, Xizhou and Hu, Xiaowei and Lu, Tong and Lu, Lewei and Li, Hongsheng and others},
  journal={arXiv preprint arXiv:2211.05778},
  year={2022}
}
```

## Contact

For questions about this evaluation framework, please open an issue in the repository.
