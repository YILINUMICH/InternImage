# InternImage DCNv3 Evaluation Project Guide

## Project Overview

This project systematically evaluates InternImage's DCNv3 operator claims for autonomous driving perception tasks. We focus on validating whether DCNv3's adaptive receptive fields actually improve detection of small, distant objects in real-world scenarios.

## Research Questions

1. **Does DCNv3 improve distant object detection?**
   - Hypothesis: DCNv3's adaptive receptive fields should improve AP for objects >50m
   - Test: Compare AP across distance bins (0-30m, 30-50m, 50-100m, >100m)

2. **Does DCNv3 improve pedestrian safety metrics?**
   - Hypothesis: Better distant detection → lower pedestrian miss rate
   - Test: Measure miss rate for pedestrians at various distances
   - Target: <10% miss rate at all distances

3. **Can DCNv3 achieve real-time performance?**
   - Hypothesis: DCNv3 is computationally efficient enough for deployment
   - Test: Measure FPS on standard GPUs (RTX 3090, A100)
   - Target: ≥30 FPS for real-time perception

4. **Does DCNv3 generalize across datasets?**
   - Hypothesis: Adaptive receptive fields improve cross-domain performance
   - Test: Train on Waymo, test on nuScenes (and vice versa)
   - Metric: Performance degradation vs. within-dataset evaluation

## Experimental Design

### Phase 1: Within-Dataset Evaluation (Weeks 1-2)

**Waymo Evaluation**
```bash
python evaluation_framework/scripts/run_waymo_eval.py \
    --config evaluation_framework/configs/waymo_experiments.yaml \
    --output evaluation_framework/results/waymo
```

**nuScenes Evaluation**
```bash
python evaluation_framework/scripts/run_nuscenes_eval.py \
    --config evaluation_framework/configs/nuscenes_experiments.yaml \
    --output evaluation_framework/results/nuscenes
```

**Models to Test:**
- InternImage-T (DCNv3 backbone)
- InternImage-S (DCNv3 backbone)
- InternImage-B (DCNv3 backbone)
- ResNet-50 (baseline, standard CNN)

**Metrics:**
- Overall: mAP, AP50, AP75
- Distance-based: AP by distance bins
- Class-specific: Pedestrian recall, cyclist recall
- Safety: Miss rate for pedestrians

### Phase 2: Baseline Comparisons (Week 3)

Compare DCNv3 against multiple baselines:
```bash
python evaluation_framework/scripts/run_baseline_comparison.py \
    --models internimage_s resnet50 swin_t convnext_t \
    --dataset waymo \
    --output evaluation_framework/results/baselines
```

**Baselines:**
1. ResNet-50 (standard CNN)
2. Swin-T (vision transformer)
3. ConvNeXt-T (modern CNN)

**Analysis:**
- Which architecture performs best at different distances?
- Does DCNv3 have a specific advantage regime (e.g., >50m)?
- What is the accuracy vs. speed trade-off?

### Phase 3: FPS Benchmarking (Week 3)

Measure real-time performance:
```bash
python evaluation_framework/scripts/measure_fps.py \
    --models all \
    --batch-sizes 1 2 4 8 \
    --gpus rtx3090 a100 \
    --output evaluation_framework/benchmarks
```

**Tests:**
- Various batch sizes (1, 2, 4, 8)
- Different input resolutions
- GPU memory usage
- Identify computational bottlenecks

**Target:** ≥30 FPS on RTX 3090 with batch size 1

### Phase 4: Cross-Dataset Generalization (Week 4)

Test domain transfer:
```bash
python evaluation_framework/scripts/cross_dataset_test.py \
    --train-dataset waymo \
    --test-dataset nuscenes \
    --models internimage_s resnet50 \
    --output evaluation_framework/results/cross_dataset
```

**Tests:**
1. Waymo → nuScenes
2. nuScenes → Waymo

**Analysis:**
- Performance drop across datasets
- Does DCNv3 generalize better than standard CNNs?
- Camera model sensitivity

### Phase 5: Analysis & Reporting (Week 5)

Generate comprehensive analysis:
```bash
# Distance analysis
python evaluation_framework/analysis/distance_analysis.py \
    --results evaluation_framework/results/ \
    --output evaluation_framework/results/analysis

# Pedestrian safety analysis
python evaluation_framework/analysis/pedestrian_analysis.py \
    --results evaluation_framework/results/ \
    --output evaluation_framework/results/analysis

# FPS analysis
python evaluation_framework/analysis/fps_analysis.py \
    --results evaluation_framework/benchmarks/ \
    --output evaluation_framework/results/analysis

# Cross-dataset analysis
python evaluation_framework/analysis/cross_dataset_analysis.py \
    --results evaluation_framework/results/cross_dataset/ \
    --output evaluation_framework/results/analysis

# Generate final report with visualizations
python evaluation_framework/analysis/visualize_results.py \
    --input evaluation_framework/results/ \
    --output evaluation_framework/reports/
```

## Expected Results

### If DCNv3 Claims Hold:

1. **Distance Performance:**
   - DCNv3 shows 5-10% higher AP for objects >50m vs. ResNet-50
   - Performance gap widens with increasing distance
   - Small object detection (AP_small) improved by 3-5%

2. **Pedestrian Safety:**
   - Miss rate <10% at all distances
   - Better performance than baselines at >40m
   - Reduced false negatives for partially occluded pedestrians

3. **Real-time Performance:**
   - InternImage-T: 35-45 FPS (meets real-time)
   - InternImage-S: 25-35 FPS (borderline real-time)
   - InternImage-B: 15-25 FPS (too slow for real-time)

4. **Generalization:**
   - <15% performance drop in cross-dataset evaluation
   - Better than ResNet-50 in domain transfer

### If Claims Need Qualification:

1. **Distance Performance:**
   - Improvement exists but is marginal (1-3% AP)
   - Only significant at extreme distances (>100m)
   - Not consistent across all classes

2. **Pedestrian Safety:**
   - Similar miss rate to baselines
   - Advantage only in specific scenarios (e.g., highway scenes)
   - Trade-off: fewer false negatives but more false positives

3. **Real-time Performance:**
   - Too slow for 30 FPS target
   - Requires optimization (TensorRT, quantization)
   - Memory usage exceeds typical embedded GPUs

4. **Generalization:**
   - High domain gap (>20% performance drop)
   - Camera-specific tuning required
   - No clear advantage over baselines

## Data Analysis Checklist

### Quantitative Analysis
- [ ] Overall mAP for each model on each dataset
- [ ] AP by distance bins (4-5 bins)
- [ ] AP by object size (small, medium, large)
- [ ] Per-class AP (especially pedestrian, cyclist)
- [ ] Pedestrian miss rate by distance
- [ ] FPS measurements (multiple batch sizes, resolutions)
- [ ] GPU memory usage
- [ ] Cross-dataset performance matrix

### Qualitative Analysis
- [ ] Visualize successful detections at various distances
- [ ] Analyze failure cases (missed detections)
- [ ] Compare predictions: DCNv3 vs. baseline
- [ ] Identify scenarios where DCNv3 excels
- [ ] Identify scenarios where DCNv3 struggles

### Statistical Validation
- [ ] Statistical significance tests (t-tests for mAP differences)
- [ ] Confidence intervals for performance metrics
- [ ] Variance analysis across test samples
- [ ] Ablation study: DCNv3 vs. standard convolution

## Deliverables

### 1. Evaluation Results
- Raw detection results (JSON format)
- Performance metrics tables (CSV)
- Benchmark measurements (JSON)

### 2. Analysis Reports
- Distance-based performance report
- Pedestrian safety analysis
- FPS benchmark report
- Cross-dataset generalization report
- Executive summary

### 3. Visualizations
- AP by distance plots (line charts)
- Miss rate comparison (bar charts)
- Precision-recall curves
- FPS comparison charts
- Qualitative detection visualizations

### 4. Documentation
- Experimental methodology
- Reproducibility instructions
- Code documentation
- Dataset preparation guide

## Reproducibility

All experiments are designed to be fully reproducible:

1. **Fixed Random Seeds**
   - Set seeds for PyTorch, NumPy, Python random
   - Document in config files

2. **Environment Documentation**
   - requirements.txt with exact versions
   - Docker container (optional)
   - Hardware specifications

3. **Dataset Versions**
   - Waymo Open Dataset v1.4.2
   - nuScenes v1.0-trainval
   - Document any preprocessing

4. **Model Checkpoints**
   - Use official pre-trained models
   - Document checkpoint URLs
   - Provide config files

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size to 1
   - Decrease input resolution
   - Use gradient checkpointing

2. **DCNv3 Compilation Fails**
   - Check CUDA/PyTorch version compatibility
   - Verify nvcc is in PATH
   - Try pre-compiled .whl files

3. **Dataset Loading Errors**
   - Verify dataset path in config
   - Check file permissions
   - Ensure dataset format is correct

4. **Slow Evaluation**
   - Enable multi-GPU evaluation
   - Use fewer test samples for debugging
   - Profile code to find bottlenecks

## Next Steps After Evaluation

### If Results are Positive:
1. Optimize for deployment (TensorRT, quantization)
2. Test on additional datasets (KITTI, Argoverse)
3. Investigate specific architectural components
4. Write technical report/paper

### If Results are Mixed:
1. Conduct ablation studies
2. Analyze failure modes in detail
3. Test architectural variations
4. Compare with additional baselines

### If Results are Negative:
1. Document limitations clearly
2. Identify where claims don't hold
3. Suggest improvements or alternative approaches
4. Provide honest assessment for research community

## Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | Environment setup, data preparation | Setup complete, data ready |
| 2 | Waymo & nuScenes evaluation | Raw results, initial metrics |
| 3 | Baseline comparison, FPS benchmarking | Comparison tables, FPS data |
| 4 | Cross-dataset tests, deep analysis | Transfer learning results |
| 5 | Report generation, visualization | Final report, all plots |

## Contact & Support

For issues or questions:
1. Check the README files in each subdirectory
2. Review the InternImage GitHub issues
3. Consult the original paper for methodology details

## Citation

If this evaluation framework is useful for your research:

```bibtex
@misc{internimage_evaluation_2024,
  title={DCNv3 Evaluation Framework for Autonomous Driving},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/YILINUMICH/InternImage}}
}
```
