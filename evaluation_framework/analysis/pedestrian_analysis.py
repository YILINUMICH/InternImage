"""
Pedestrian Detection Analysis - Safety-Critical Metric

Focuses specifically on pedestrian detection performance as the primary safety metric
for autonomous driving. Analyzes:
1. Pedestrian detection recall at various distances
2. Miss rate analysis (critical for safety)
3. False positive analysis
4. Day/night performance (if available)
5. Occlusion handling
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

class PedestrianAnalyzer:
    def __init__(self, results_dir: str, output_dir: str = None):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / "pedestrian_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Distance bins for safety analysis
        self.safety_distance_bins = [
            (0, 10, "Critical (<10m)"),
            (10, 20, "Near (10-20m)"),
            (20, 40, "Medium (20-40m)"),
            (40, 80, "Far (40-80m)"),
            (80, float('inf'), "Very Far (>80m)")
        ]
        
        # Occlusion levels
        self.occlusion_levels = ["none", "partial", "heavy"]
        
    def filter_pedestrian_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter only pedestrian class detections"""
        pedestrian_classes = ['pedestrian', 'person', 'human']
        
        return [
            det for det in detections 
            if det.get('class', '').lower() in pedestrian_classes
        ]
    
    def calculate_pedestrian_recall(self, detections: List[Dict], 
                                   ground_truth: List[Dict],
                                   iou_threshold: float = 0.5) -> Dict:
        """Calculate pedestrian recall metrics"""
        if not ground_truth:
            return {'recall': 0.0, 'precision': 0.0, 'f1': 0.0}
        
        # Match detections to GT
        matched_gt = set()
        true_positives = 0
        
        for det in detections:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(ground_truth):
                if i in matched_gt:
                    continue
                
                iou = self._calculate_iou(det.get('bbox', []), gt.get('bbox', []))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                true_positives += 1
                matched_gt.add(best_gt_idx)
        
        # Calculate metrics
        recall = true_positives / len(ground_truth) if ground_truth else 0
        precision = true_positives / len(detections) if detections else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'true_positives': true_positives,
            'false_negatives': len(ground_truth) - true_positives,
            'false_positives': len(detections) - true_positives
        }
    
    def calculate_miss_rate(self, detections: List[Dict], 
                          ground_truth: List[Dict],
                          distance_range: Tuple[float, float] = None) -> float:
        """
        Calculate miss rate (critical safety metric)
        Miss rate = False Negatives / (True Positives + False Negatives)
        """
        if distance_range:
            min_d, max_d = distance_range
            ground_truth = [
                gt for gt in ground_truth 
                if min_d <= gt.get('distance', 0) < max_d
            ]
            detections = [
                det for det in detections 
                if min_d <= det.get('distance', 0) < max_d
            ]
        
        metrics = self.calculate_pedestrian_recall(detections, ground_truth)
        miss_rate = metrics['false_negatives'] / len(ground_truth) if ground_truth else 0
        
        return miss_rate
    
    def analyze_by_distance(self, results: Dict) -> pd.DataFrame:
        """Analyze pedestrian detection by distance bins"""
        data = []
        
        for model_name, model_results in results.items():
            all_dets = model_results.get('detections', [])
            all_gt = model_results.get('ground_truth', [])
            
            # Filter pedestrians
            ped_dets = self.filter_pedestrian_detections(all_dets)
            ped_gt = self.filter_pedestrian_detections(all_gt)
            
            for min_d, max_d, label in self.safety_distance_bins:
                # Filter by distance
                dist_dets = [d for d in ped_dets if min_d <= d.get('distance', 0) < max_d]
                dist_gt = [d for d in ped_gt if min_d <= d.get('distance', 0) < max_d]
                
                if not dist_gt:
                    continue
                
                # Calculate metrics
                metrics = self.calculate_pedestrian_recall(dist_dets, dist_gt)
                miss_rate = self.calculate_miss_rate(dist_dets, dist_gt)
                
                data.append({
                    'model': model_name,
                    'distance_range': label,
                    'recall': metrics['recall'],
                    'precision': metrics['precision'],
                    'f1': metrics['f1'],
                    'miss_rate': miss_rate,
                    'num_pedestrians': len(dist_gt),
                    'detected': metrics['true_positives'],
                    'missed': metrics['false_negatives']
                })
        
        return pd.DataFrame(data)
    
    def analyze_by_occlusion(self, results: Dict) -> pd.DataFrame:
        """Analyze performance under different occlusion levels"""
        data = []
        
        for model_name, model_results in results.items():
            all_dets = model_results.get('detections', [])
            all_gt = model_results.get('ground_truth', [])
            
            ped_dets = self.filter_pedestrian_detections(all_dets)
            ped_gt = self.filter_pedestrian_detections(all_gt)
            
            for occlusion_level in self.occlusion_levels:
                # Filter by occlusion
                occ_gt = [d for d in ped_gt 
                         if d.get('occlusion', 'none') == occlusion_level]
                
                if not occ_gt:
                    continue
                
                # Match detections to this subset
                occ_dets = []
                for det in ped_dets:
                    for gt in occ_gt:
                        if self._calculate_iou(det.get('bbox', []), 
                                             gt.get('bbox', [])) > 0.3:
                            occ_dets.append(det)
                            break
                
                metrics = self.calculate_pedestrian_recall(occ_dets, occ_gt)
                
                data.append({
                    'model': model_name,
                    'occlusion': occlusion_level,
                    'recall': metrics['recall'],
                    'precision': metrics['precision'],
                    'num_pedestrians': len(occ_gt)
                })
        
        return pd.DataFrame(data)
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
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
    
    def plot_miss_rate_by_distance(self, df: pd.DataFrame, save_path: str = None):
        """Plot miss rate across distances (critical safety visualization)"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model].sort_values('distance_range')
            ax.plot(model_data['distance_range'], model_data['miss_rate'] * 100, 
                   marker='o', linewidth=2.5, label=model, markersize=8)
        
        # Add safety threshold line
        ax.axhline(y=10, color='red', linestyle='--', linewidth=2, 
                  label='10% Target', alpha=0.7)
        
        ax.set_xlabel('Distance Range', fontsize=13, fontweight='bold')
        ax.set_ylabel('Miss Rate (%)', fontsize=13, fontweight='bold')
        ax.set_title('Pedestrian Miss Rate by Distance (Safety Critical)', 
                    fontsize=15, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Rotate x labels
        plt.xticks(rotation=15, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "miss_rate_by_distance.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_recall_comparison(self, df: pd.DataFrame, save_path: str = None):
        """Compare recall across models and distances"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Pivot data for grouped bar chart
        pivot_data = df.pivot(index='distance_range', columns='model', values='recall')
        
        pivot_data.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_xlabel('Distance Range', fontsize=13, fontweight='bold')
        ax.set_ylabel('Recall', fontsize=13, fontweight='bold')
        ax.set_title('Pedestrian Detection Recall by Distance', 
                    fontsize=15, fontweight='bold')
        ax.legend(title='Model', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels
        plt.xticks(rotation=15, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "recall_comparison.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_precision_recall_curve(self, detections: List[Dict], 
                                   ground_truth: List[Dict],
                                   model_name: str, save_path: str = None):
        """Plot precision-recall curve for pedestrian detection"""
        # Sort detections by confidence
        sorted_dets = sorted(detections, key=lambda x: x.get('score', 0), reverse=True)
        
        precisions = []
        recalls = []
        
        for threshold in np.linspace(0, 1, 50):
            # Filter by threshold
            filtered_dets = [d for d in sorted_dets if d.get('score', 0) >= threshold]
            
            metrics = self.calculate_pedestrian_recall(filtered_dets, ground_truth)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(recalls, precisions, linewidth=2.5, color='blue')
        ax.fill_between(recalls, precisions, alpha=0.3, color='blue')
        
        ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
        ax.set_title(f'Precision-Recall Curve - {model_name}\n(Pedestrian Detection)', 
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / f"pr_curve_{model_name}.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def generate_safety_report(self, distance_df: pd.DataFrame, 
                              occlusion_df: pd.DataFrame = None) -> str:
        """Generate pedestrian safety report"""
        report = []
        report.append("="*70)
        report.append("Pedestrian Detection Safety Analysis Report")
        report.append("="*70)
        report.append("")
        
        report.append("🚨 CRITICAL SAFETY METRICS")
        report.append("-" * 40)
        report.append("")
        
        # Overall pedestrian recall
        report.append("1. Overall Pedestrian Recall by Model")
        overall_recall = distance_df.groupby('model')['recall'].mean()
        for model, recall in overall_recall.items():
            status = "✓ GOOD" if recall > 0.9 else "⚠ NEEDS IMPROVEMENT"
            report.append(f"  • {model}: {recall:.3f} ({recall*100:.1f}%) {status}")
        report.append("")
        
        # Miss rate analysis (CRITICAL)
        report.append("2. Miss Rate by Distance (Safety Critical)")
        report.append("   Target: <10% miss rate for all distances")
        report.append("")
        
        for model in distance_df['model'].unique():
            report.append(f"   {model}:")
            model_data = distance_df[distance_df['model'] == model]
            
            for _, row in model_data.iterrows():
                miss_pct = row['miss_rate'] * 100
                status = "✓" if miss_pct < 10 else "✗"
                report.append(f"     {status} {row['distance_range']}: "
                            f"{miss_pct:.1f}% miss rate "
                            f"({row['missed']}/{row['num_pedestrians']} missed)")
            report.append("")
        
        # Critical distance performance
        report.append("3. Critical Distance (<10m) Performance")
        critical_data = distance_df[distance_df['distance_range'].str.contains('Critical')]
        
        if not critical_data.empty:
            for _, row in critical_data.iterrows():
                report.append(f"  • {row['model']}:")
                report.append(f"    - Recall: {row['recall']:.3f} ({row['recall']*100:.1f}%)")
                report.append(f"    - Miss Rate: {row['miss_rate']*100:.1f}%")
                report.append(f"    - Missed: {row['missed']} pedestrians")
        report.append("")
        
        # Occlusion analysis
        if occlusion_df is not None and not occlusion_df.empty:
            report.append("4. Performance Under Occlusion")
            report.append("-" * 40)
            
            for model in occlusion_df['model'].unique():
                report.append(f"  {model}:")
                model_occ = occlusion_df[occlusion_df['model'] == model]
                
                for _, row in model_occ.iterrows():
                    report.append(f"    • {row['occlusion'].capitalize()}: "
                                f"{row['recall']:.3f} recall")
            report.append("")
        
        # Recommendations
        report.append("5. Safety Recommendations")
        report.append("-" * 40)
        
        # Find worst performing scenarios
        worst_recall = distance_df.loc[distance_df['recall'].idxmin()]
        report.append(f"  • Lowest recall: {worst_recall['model']} at "
                     f"{worst_recall['distance_range']} ({worst_recall['recall']:.3f})")
        
        highest_miss = distance_df.loc[distance_df['miss_rate'].idxmax()]
        report.append(f"  • Highest miss rate: {highest_miss['model']} at "
                     f"{highest_miss['distance_range']} ({highest_miss['miss_rate']*100:.1f}%)")
        
        report.append("")
        report.append("  ⚠ Areas requiring attention:")
        
        # Identify problem areas
        problem_areas = distance_df[distance_df['miss_rate'] > 0.1]
        if not problem_areas.empty:
            for _, row in problem_areas.iterrows():
                report.append(f"    - {row['model']}, {row['distance_range']}: "
                            f"{row['miss_rate']*100:.1f}% miss rate")
        else:
            report.append("    - No critical issues found (all miss rates <10%)")
        
        report.append("")
        report.append("="*70)
        
        return "\n".join(report)
    
    def run_full_analysis(self, result_files: Dict[str, str]):
        """Run complete pedestrian safety analysis"""
        print("Running pedestrian safety analysis...")
        
        # Load results
        results = {}
        for model_name, file_path in result_files.items():
            print(f"  Loading {model_name} results...")
            with open(file_path, 'r') as f:
                results[model_name] = json.load(f)
        
        # Distance-based analysis
        print("  Analyzing by distance...")
        distance_df = self.analyze_by_distance(results)
        
        # Occlusion analysis
        print("  Analyzing by occlusion...")
        occlusion_df = self.analyze_by_occlusion(results)
        
        # Generate plots
        print("  Generating visualizations...")
        self.plot_miss_rate_by_distance(distance_df)
        self.plot_recall_comparison(distance_df)
        
        # PR curves for each model
        for model_name, model_results in results.items():
            dets = self.filter_pedestrian_detections(model_results.get('detections', []))
            gt = self.filter_pedestrian_detections(model_results.get('ground_truth', []))
            
            if dets and gt:
                self.plot_precision_recall_curve(dets, gt, model_name)
        
        # Generate safety report
        print("  Generating safety report...")
        report = self.generate_safety_report(distance_df, occlusion_df)
        
        # Save reports
        report_path = self.output_dir / "pedestrian_safety_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        distance_df.to_csv(self.output_dir / "pedestrian_distance_metrics.csv", index=False)
        if not occlusion_df.empty:
            occlusion_df.to_csv(self.output_dir / "pedestrian_occlusion_metrics.csv", index=False)
        
        print(f"\n✓ Pedestrian analysis complete! Results saved to {self.output_dir}")
        print(f"  - Plots: {self.output_dir}/*.png")
        print(f"  - Report: {report_path}")
        print(f"  - Data: {self.output_dir}/*.csv")
        
        return distance_df, occlusion_df, report

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze pedestrian detection performance (safety-critical)"
    )
    parser.add_argument("--results", type=str, required=True,
                       help="Directory containing result files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for analysis")
    parser.add_argument("--models", type=str, nargs='+',
                       help="List of model result files")
    
    args = parser.parse_args()
    
    analyzer = PedestrianAnalyzer(args.results, args.output)
    
    # Load result files
    if args.models:
        result_files = {Path(f).stem: f for f in args.models}
    else:
        result_files = {f.stem: str(f) for f in Path(args.results).glob("*.json")}
    
    if not result_files:
        print("No result files found!")
        return
    
    # Run analysis
    analyzer.run_full_analysis(result_files)

if __name__ == "__main__":
    main()
