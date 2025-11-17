"""
Distance-based Detection Analysis

Analyzes object detection performance across different distance bins to verify
DCNv3's claim of improved distant object detection through adaptive receptive fields.

Key analyses:
1. AP by distance bins (near, medium, far)
2. DCNv3 vs baseline comparison at each distance
3. Small object detection performance
4. Visualization of distance-dependent improvements
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple

class DistanceAnalyzer:
    def __init__(self, results_dir: str, output_dir: str = None):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Distance bins (meters)
        self.distance_bins = [(0, 30), (30, 50), (50, 100), (100, float('inf'))]
        self.distance_labels = ['0-30m', '30-50m', '50-100m', '>100m']
        
        # Size bins (pixels)
        self.size_bins = [(0, 32**2), (32**2, 96**2), (96**2, float('inf'))]
        self.size_labels = ['Small (<32²)', 'Medium (32²-96²)', 'Large (>96²)']
        
    def load_results(self, result_file: str) -> Dict:
        """Load detection results from JSON file"""
        with open(result_file, 'r') as f:
            return json.load(f)
    
    def bin_detections_by_distance(self, detections: List[Dict]) -> Dict:
        """Bin detections by distance ranges"""
        binned = {label: [] for label in self.distance_labels}
        
        for det in detections:
            if 'distance' not in det:
                continue
                
            distance = det['distance']
            
            for (min_d, max_d), label in zip(self.distance_bins, self.distance_labels):
                if min_d <= distance < max_d:
                    binned[label].append(det)
                    break
        
        return binned
    
    def bin_detections_by_size(self, detections: List[Dict]) -> Dict:
        """Bin detections by object size"""
        binned = {label: [] for label in self.size_labels}
        
        for det in detections:
            if 'bbox' not in det:
                continue
            
            # Calculate bbox area
            bbox = det['bbox']  # [x1, y1, x2, y2]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            for (min_s, max_s), label in zip(self.size_bins, self.size_labels):
                if min_s <= area < max_s:
                    binned[label].append(det)
                    break
        
        return binned
    
    def calculate_ap_by_distance(self, results: Dict) -> pd.DataFrame:
        """Calculate Average Precision for each distance bin"""
        data = []
        
        for model_name, model_results in results.items():
            detections = model_results.get('detections', [])
            ground_truth = model_results.get('ground_truth', [])
            
            # Bin detections and GT
            det_binned = self.bin_detections_by_distance(detections)
            gt_binned = self.bin_detections_by_distance(ground_truth)
            
            for label in self.distance_labels:
                # Calculate AP for this bin
                ap = self._calculate_ap(det_binned[label], gt_binned[label])
                
                data.append({
                    'model': model_name,
                    'distance_range': label,
                    'ap': ap,
                    'num_detections': len(det_binned[label]),
                    'num_gt': len(gt_binned[label])
                })
        
        return pd.DataFrame(data)
    
    def _calculate_ap(self, detections: List[Dict], ground_truth: List[Dict], 
                     iou_threshold: float = 0.5) -> float:
        """Calculate Average Precision for a set of detections"""
        if not ground_truth:
            return 0.0
        
        # Sort detections by confidence
        detections = sorted(detections, key=lambda x: x.get('score', 0), reverse=True)
        
        # Match detections to ground truth
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        matched_gt = set()
        
        for i, det in enumerate(detections):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(ground_truth):
                if j in matched_gt:
                    continue
                
                iou = self._calculate_iou(det.get('bbox', []), gt.get('bbox', []))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[i] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(ground_truth)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.linspace(0, 1, 11):
            p = precisions[recalls >= t]
            ap += np.max(p) if len(p) > 0 else 0
        
        return ap / 11
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        if not bbox1 or not bbox2:
            return 0.0
        
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-10)
    
    def compare_models_by_distance(self, ap_df: pd.DataFrame) -> pd.DataFrame:
        """Compare multiple models across distance bins"""
        # Pivot to get models as columns
        comparison = ap_df.pivot(index='distance_range', 
                                columns='model', 
                                values='ap')
        
        # Calculate improvement over baseline (if exists)
        if 'baseline' in comparison.columns:
            for col in comparison.columns:
                if col != 'baseline':
                    comparison[f'{col}_improvement'] = \
                        (comparison[col] - comparison['baseline']) / comparison['baseline'] * 100
        
        return comparison
    
    def plot_ap_by_distance(self, ap_df: pd.DataFrame, save_path: str = None):
        """Plot AP across distance ranges for different models"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot lines for each model
        for model in ap_df['model'].unique():
            model_data = ap_df[ap_df['model'] == model]
            ax.plot(model_data['distance_range'], model_data['ap'], 
                   marker='o', linewidth=2, label=model)
        
        ax.set_xlabel('Distance Range', fontsize=12)
        ax.set_ylabel('Average Precision (AP)', fontsize=12)
        ax.set_title('Detection Performance by Distance Range', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.savefig(self.output_dir / "ap_by_distance.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_improvement_heatmap(self, comparison_df: pd.DataFrame, save_path: str = None):
        """Plot heatmap of improvements over baseline"""
        # Filter improvement columns
        improvement_cols = [col for col in comparison_df.columns if '_improvement' in col]
        
        if not improvement_cols:
            print("No improvement data available")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create heatmap data
        heatmap_data = comparison_df[improvement_cols]
        heatmap_data.columns = [col.replace('_improvement', '') for col in improvement_cols]
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=0, cbar_kws={'label': 'Improvement (%)'}, ax=ax)
        
        ax.set_title('Performance Improvement Over Baseline by Distance', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Distance Range', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "improvement_heatmap.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_detection_distribution(self, results: Dict, save_path: str = None):
        """Plot distribution of detections across distances"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for ax, (model_name, model_results) in zip(axes.flat, results.items()):
            detections = model_results.get('detections', [])
            distances = [d['distance'] for d in detections if 'distance' in d]
            
            if distances:
                ax.hist(distances, bins=50, alpha=0.7, edgecolor='black')
                ax.axvline(x=30, color='r', linestyle='--', label='30m')
                ax.axvline(x=50, color='orange', linestyle='--', label='50m')
                ax.axvline(x=100, color='y', linestyle='--', label='100m')
                
                ax.set_xlabel('Distance (m)', fontsize=12)
                ax.set_ylabel('Number of Detections', fontsize=12)
                ax.set_title(f'{model_name} - Detection Distribution', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "detection_distribution.png", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def generate_summary_report(self, ap_df: pd.DataFrame, comparison_df: pd.DataFrame) -> str:
        """Generate text summary of distance analysis"""
        report = []
        report.append("="*70)
        report.append("Distance-based Detection Analysis Report")
        report.append("="*70)
        report.append("")
        
        # Overall statistics
        report.append("1. Overall Performance by Distance")
        report.append("-" * 40)
        report.append(ap_df.groupby('distance_range')['ap'].agg(['mean', 'std']).to_string())
        report.append("")
        
        # Model comparison
        report.append("2. Model Comparison")
        report.append("-" * 40)
        report.append(comparison_df.to_string())
        report.append("")
        
        # Key findings
        report.append("3. Key Findings")
        report.append("-" * 40)
        
        # Find best performing model at each distance
        for dist in self.distance_labels:
            dist_data = ap_df[ap_df['distance_range'] == dist]
            best_model = dist_data.loc[dist_data['ap'].idxmax(), 'model']
            best_ap = dist_data['ap'].max()
            report.append(f"  • Best at {dist}: {best_model} (AP={best_ap:.3f})")
        
        report.append("")
        
        # Performance trends
        report.append("4. Performance Trends")
        report.append("-" * 40)
        
        for model in ap_df['model'].unique():
            model_data = ap_df[ap_df['model'] == model].sort_values('distance_range')
            aps = model_data['ap'].values
            
            if len(aps) > 1:
                trend = "declining" if aps[-1] < aps[0] else "improving"
                drop = (aps[0] - aps[-1]) / aps[0] * 100
                report.append(f"  • {model}: {trend} ({drop:.1f}% change from near to far)")
        
        report.append("")
        report.append("="*70)
        
        return "\n".join(report)
    
    def run_full_analysis(self, result_files: Dict[str, str]):
        """Run complete distance-based analysis"""
        print("Running distance-based analysis...")
        
        # Load all results
        results = {}
        for model_name, file_path in result_files.items():
            print(f"  Loading {model_name} results...")
            results[model_name] = self.load_results(file_path)
        
        # Calculate AP by distance
        print("  Calculating AP by distance...")
        ap_df = self.calculate_ap_by_distance(results)
        
        # Compare models
        print("  Comparing models...")
        comparison_df = self.compare_models_by_distance(ap_df)
        
        # Generate plots
        print("  Generating visualizations...")
        self.plot_ap_by_distance(ap_df)
        self.plot_improvement_heatmap(comparison_df)
        self.plot_detection_distribution(results)
        
        # Generate report
        print("  Generating summary report...")
        report = self.generate_summary_report(ap_df, comparison_df)
        
        # Save report
        report_path = self.output_dir / "distance_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Analysis complete! Results saved to {self.output_dir}")
        print(f"  - Plots: {self.output_dir}")
        print(f"  - Report: {report_path}")
        
        return ap_df, comparison_df, report

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze detection performance by distance")
    parser.add_argument("--results", type=str, required=True,
                       help="Directory containing result files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for analysis")
    parser.add_argument("--models", type=str, nargs='+',
                       help="List of model result files")
    
    args = parser.parse_args()
    
    analyzer = DistanceAnalyzer(args.results, args.output)
    
    # Example: Load results from specified files
    if args.models:
        result_files = {
            Path(f).stem: f for f in args.models
        }
    else:
        # Auto-discover result files
        result_files = {
            f.stem: str(f) 
            for f in Path(args.results).glob("*.json")
        }
    
    if not result_files:
        print("No result files found!")
        return
    
    # Run analysis
    analyzer.run_full_analysis(result_files)

if __name__ == "__main__":
    main()
