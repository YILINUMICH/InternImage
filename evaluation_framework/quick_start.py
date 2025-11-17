"""
Quick Start Script for InternImage DCNv3 Evaluation

This script provides a simple interface to run the entire evaluation pipeline
with sensible defaults. Perfect for getting started quickly.

Usage:
    python quick_start.py --dataset waymo --model internimage_s
    python quick_start.py --dataset nuscenes --model all --run-analysis
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

class QuickStart:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.eval_framework = Path(__file__).parent
        
    def check_environment(self):
        """Check if environment is properly set up"""
        print("🔍 Checking environment...")
        
        checks = {
            "PyTorch": self._check_pytorch(),
            "CUDA": self._check_cuda(),
            "DCNv3": self._check_dcnv3(),
            "Datasets": self._check_datasets(),
            "Models": self._check_models()
        }
        
        all_good = all(checks.values())
        
        for name, status in checks.items():
            symbol = "✓" if status else "✗"
            print(f"  {symbol} {name}")
        
        if not all_good:
            print("\n⚠ Some components are missing. Run setup_environment.py first!")
            return False
        
        print("✓ Environment ready!\n")
        return True
    
    def _check_pytorch(self):
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def _check_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _check_dcnv3(self):
        dcnv3_path = self.project_root / "detection" / "ops_dcnv3"
        return dcnv3_path.exists()
    
    def _check_datasets(self):
        data_dir = self.eval_framework / "data"
        return data_dir.exists()
    
    def _check_models(self):
        model_dir = self.eval_framework / "models" / "pretrained"
        if not model_dir.exists():
            return False
        # Check if any .pth files exist
        return len(list(model_dir.glob("*.pth"))) > 0
    
    def run_evaluation(self, dataset, model, skip_analysis=False):
        """Run evaluation pipeline"""
        
        print(f"🚀 Starting evaluation: {dataset} dataset, {model} model\n")
        
        # Determine config file
        config_file = self.eval_framework / "configs" / f"{dataset}_experiments.yaml"
        
        if not config_file.exists():
            print(f"✗ Config file not found: {config_file}")
            return False
        
        # Determine script
        if dataset == "waymo":
            script = self.eval_framework / "scripts" / "run_waymo_eval.py"
        elif dataset == "nuscenes":
            script = self.eval_framework / "scripts" / "run_nuscenes_eval.py"
        else:
            print(f"✗ Unknown dataset: {dataset}")
            return False
        
        if not script.exists():
            print(f"⚠ Evaluation script not implemented yet: {script}")
            print("  This is a framework template. You'll need to implement the dataset-specific code.")
            return False
        
        # Run evaluation
        print(f"📊 Running evaluation...")
        print(f"  Config: {config_file}")
        print(f"  Script: {script}")
        print(f"  Model: {model}")
        print()
        
        cmd = [
            sys.executable,
            str(script),
            "--config", str(config_file),
            "--model", model,
            "--output", str(self.eval_framework / "results" / dataset)
        ]
        
        try:
            result = subprocess.run(cmd, check=True)
            print("\n✓ Evaluation complete!")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Evaluation failed: {e}")
            return False
        
        # Run analysis
        if not skip_analysis:
            print("\n📈 Running analysis...")
            self.run_analysis(dataset)
        
        return True
    
    def run_analysis(self, dataset):
        """Run all analysis scripts"""
        
        results_dir = self.eval_framework / "results" / dataset
        
        if not results_dir.exists():
            print(f"✗ Results directory not found: {results_dir}")
            return False
        
        analyses = [
            ("distance_analysis.py", "Distance-based analysis"),
            ("pedestrian_analysis.py", "Pedestrian safety analysis"),
            ("fps_analysis.py", "FPS benchmarking")
        ]
        
        for script_name, description in analyses:
            script = self.eval_framework / "analysis" / script_name
            
            if not script.exists():
                print(f"  ⚠ {script_name} not found, skipping...")
                continue
            
            print(f"  Running {description}...")
            
            cmd = [
                sys.executable,
                str(script),
                "--results", str(results_dir),
                "--output", str(results_dir / "analysis")
            ]
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                print(f"    ✓ {description} complete")
            except subprocess.CalledProcessError as e:
                print(f"    ✗ {description} failed: {e}")
        
        print("\n✓ All analysis complete!")
        print(f"📁 Results saved to: {results_dir / 'analysis'}")
    
    def print_quick_commands(self):
        """Print helpful quick start commands"""
        print("\n" + "="*70)
        print("Quick Start Commands")
        print("="*70)
        print()
        print("1. Setup environment:")
        print("   python evaluation_framework/scripts/setup_environment.py")
        print()
        print("2. Run Waymo evaluation:")
        print("   python evaluation_framework/quick_start.py --dataset waymo --model internimage_s")
        print()
        print("3. Run nuScenes evaluation:")
        print("   python evaluation_framework/quick_start.py --dataset nuscenes --model internimage_s")
        print()
        print("4. Run all models:")
        print("   python evaluation_framework/quick_start.py --dataset waymo --model all")
        print()
        print("5. Skip analysis:")
        print("   python evaluation_framework/quick_start.py --dataset waymo --model internimage_s --skip-analysis")
        print()
        print("6. Manual analysis:")
        print("   python evaluation_framework/analysis/distance_analysis.py --results results/waymo")
        print()
        print("="*70)

def main():
    parser = argparse.ArgumentParser(
        description="Quick start script for InternImage DCNv3 evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Waymo evaluation with InternImage-S
  python quick_start.py --dataset waymo --model internimage_s
  
  # Run nuScenes evaluation with all models
  python quick_start.py --dataset nuscenes --model all
  
  # Run evaluation without analysis
  python quick_start.py --dataset waymo --model internimage_t --skip-analysis
  
  # Just check environment
  python quick_start.py --check-only
        """
    )
    
    parser.add_argument("--dataset", type=str, choices=["waymo", "nuscenes"],
                       help="Dataset to evaluate on")
    parser.add_argument("--model", type=str, 
                       default="internimage_s",
                       choices=["internimage_t", "internimage_s", "internimage_b", 
                               "resnet50_baseline", "all"],
                       help="Model to evaluate")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip automatic analysis after evaluation")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check environment, don't run evaluation")
    parser.add_argument("--show-commands", action="store_true",
                       help="Show helpful quick start commands")
    
    args = parser.parse_args()
    
    quick_start = QuickStart()
    
    # Show commands if requested
    if args.show_commands:
        quick_start.print_quick_commands()
        return
    
    # Check environment
    if not quick_start.check_environment():
        print("\n❌ Environment check failed!")
        print("\nRun this to set up the environment:")
        print("  python evaluation_framework/scripts/setup_environment.py")
        sys.exit(1)
    
    if args.check_only:
        print("✓ Environment check passed!")
        return
    
    # Validate arguments
    if not args.dataset:
        print("✗ Please specify a dataset: --dataset {waymo,nuscenes}")
        print("\nFor help, run: python quick_start.py --help")
        sys.exit(1)
    
    # Run evaluation
    if args.model == "all":
        models = ["internimage_t", "internimage_s", "internimage_b", "resnet50_baseline"]
        
        print(f"🔄 Running evaluation for all models on {args.dataset}...")
        print(f"   Models: {', '.join(models)}\n")
        
        for model in models:
            print(f"\n{'='*70}")
            print(f"Model: {model}")
            print(f"{'='*70}\n")
            
            success = quick_start.run_evaluation(args.dataset, model, args.skip_analysis)
            
            if not success:
                print(f"\n⚠ {model} evaluation had issues, continuing to next model...")
        
        print("\n" + "="*70)
        print("✓ All evaluations complete!")
        print("="*70)
    else:
        success = quick_start.run_evaluation(args.dataset, args.model, args.skip_analysis)
        
        if not success:
            sys.exit(1)
    
    print("\n🎉 Evaluation pipeline complete!")
    print(f"\n📊 View results at: evaluation_framework/results/{args.dataset}")
    print(f"📈 View plots at: evaluation_framework/results/{args.dataset}/plots")
    print(f"📄 View reports at: evaluation_framework/results/{args.dataset}/analysis")

if __name__ == "__main__":
    main()
