"""
FPS (Frames Per Second) Benchmarking and Runtime Analysis

Measures inference speed and computational performance to determine if DCNv3
is viable for real-time autonomous driving (≥30 FPS target).

Analyzes:
1. Inference FPS on different GPUs
2. Batch size impact
3. Resolution impact
4. Memory usage
5. Computational bottlenecks
"""

import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import subprocess
import platform

class FPSBenchmark:
    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Target FPS for real-time perception
        self.target_fps = 30
        
        # Test configurations
        self.batch_sizes = [1, 2, 4, 8]
        self.input_sizes = [(800, 1280), (960, 1600), (1024, 1920)]
        
    def get_gpu_info(self) -> Dict:
        """Get GPU information"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        info = {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "capability": torch.cuda.get_device_capability(0),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
        }
        
        return info
    
    def measure_model_fps(self, model, input_size: tuple, 
                         batch_size: int = 1, 
                         num_warmup: int = 10,
                         num_iterations: int = 100) -> Dict:
        """
        Measure FPS for a model
        
        Returns:
            Dict with fps, latency, memory usage
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, *input_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
        
        # Synchronize
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure inference time
        times = []
        memory_used = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start_mem = torch.cuda.memory_allocated() / 1e9
                
                start_time = time.time()
                _ = model(dummy_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    end_mem = torch.cuda.memory_allocated() / 1e9
                    memory_used.append(end_mem)
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        fps = batch_size / times.mean()
        latency_ms = times.mean() * 1000
        
        results = {
            "fps": fps,
            "latency_ms": latency_ms,
            "latency_std_ms": times.std() * 1000,
            "batch_size": batch_size,
            "input_size": input_size,
            "gpu_memory_gb": np.mean(memory_used) if memory_used else 0,
            "meets_realtime": fps >= self.target_fps
        }
        
        return results
    
    def benchmark_model(self, model, model_name: str) -> Dict:
        """Run complete benchmark suite on a model"""
        print(f"\nBenchmarking {model_name}...")
        
        gpu_info = self.get_gpu_info()
        print(f"  GPU: {gpu_info.get('name', 'CPU only')}")
        
        all_results = {
            "model_name": model_name,
            "gpu_info": gpu_info,
            "benchmarks": []
        }
        
        # Test different configurations
        for input_size in self.input_sizes:
            for batch_size in self.batch_sizes:
                print(f"  Testing: batch_size={batch_size}, input_size={input_size}")
                
                try:
                    result = self.measure_model_fps(
                        model, 
                        input_size, 
                        batch_size=batch_size,
                        num_warmup=10,
                        num_iterations=100
                    )
                    
                    all_results["benchmarks"].append(result)
                    
                    status = "✓ Real-time" if result["meets_realtime"] else "✗ Too slow"
                    print(f"    FPS: {result['fps']:.2f}, Latency: {result['latency_ms']:.2f}ms {status}")
                    
                except RuntimeError as e:
                    print(f"    ✗ Failed: {e}")
                    continue
        
        # Save results
        output_file = self.output_dir / f"{model_name}_fps_benchmark.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"  ✓ Results saved to {output_file}")
        
        return all_results
    
    def compare_models(self, results: Dict[str, Dict]) -> None:
        """Compare FPS across multiple models"""
        print("\nGenerating comparison visualizations...")
        
        # Extract data for batch_size=1 (most common)
        comparison_data = {}
        for model_name, model_results in results.items():
            batch1_results = [
                r for r in model_results["benchmarks"] 
                if r["batch_size"] == 1
            ]
            
            if batch1_results:
                comparison_data[model_name] = {
                    "fps": [r["fps"] for r in batch1_results],
                    "latency": [r["latency_ms"] for r in batch1_results],
                    "memory": [r["gpu_memory_gb"] for r in batch1_results],
                    "input_sizes": [r["input_size"] for r in batch1_results]
                }
        
        # Plot FPS comparison
        self._plot_fps_comparison(comparison_data)
        
        # Plot latency comparison
        self._plot_latency_comparison(comparison_data)
        
        # Plot memory usage
        self._plot_memory_comparison(comparison_data)
        
        # Plot batch size scaling
        self._plot_batch_scaling(results)
    
    def _plot_fps_comparison(self, data: Dict):
        """Plot FPS comparison across models"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(data))
        width = 0.25
        
        # Get input sizes (assuming same for all models)
        input_sizes = list(data.values())[0]["input_sizes"]
        
        for i, size in enumerate(input_sizes):
            fps_values = [data[model]["fps"][i] for model in data.keys()]
            ax.bar(x + i*width, fps_values, width, 
                  label=f'{size[0]}x{size[1]}')
        
        # Add target line
        ax.axhline(y=self.target_fps, color='red', linestyle='--', 
                  linewidth=2, label=f'{self.target_fps} FPS Target')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('FPS (Frames Per Second)', fontsize=12, fontweight='bold')
        ax.set_title('Real-time Performance Comparison (Batch Size = 1)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(data.keys(), rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fps_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_latency_comparison(self, data: Dict):
        """Plot latency comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(data.keys())
        latencies = [np.mean(data[model]["latency"]) for model in models]
        
        bars = ax.bar(models, latencies, color=['green' if l < 33.3 else 'red' 
                                               for l in latencies])
        
        # Add real-time threshold
        ax.axhline(y=33.3, color='red', linestyle='--', 
                  label='33.3ms (30 FPS threshold)')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}ms',
                   ha='center', va='bottom')
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / "latency_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_comparison(self, data: Dict):
        """Plot GPU memory usage"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(data.keys())
        memory = [np.mean(data[model]["memory"]) for model in models]
        
        ax.bar(models, memory, color='steelblue')
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('GPU Memory (GB)', fontsize=12, fontweight='bold')
        ax.set_title('GPU Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / "memory_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_batch_scaling(self, results: Dict):
        """Plot how FPS scales with batch size"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for model_name, model_results in results.items():
            # Group by batch size
            batch_data = {}
            for r in model_results["benchmarks"]:
                bs = r["batch_size"]
                if bs not in batch_data:
                    batch_data[bs] = []
                batch_data[bs].append(r["fps"])
            
            batch_sizes = sorted(batch_data.keys())
            avg_fps = [np.mean(batch_data[bs]) for bs in batch_sizes]
            
            ax.plot(batch_sizes, avg_fps, marker='o', linewidth=2, 
                   label=model_name, markersize=8)
        
        ax.axhline(y=self.target_fps, color='red', linestyle='--', 
                  label=f'{self.target_fps} FPS Target')
        
        ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('FPS', fontsize=12, fontweight='bold')
        ax.set_title('Batch Size Scaling', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "batch_scaling.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, results: Dict[str, Dict]) -> str:
        """Generate text report"""
        report = []
        report.append("="*70)
        report.append("FPS Benchmark Report - Real-time Viability Analysis")
        report.append("="*70)
        report.append("")
        
        # GPU info
        first_result = list(results.values())[0]
        gpu_info = first_result.get("gpu_info", {})
        
        report.append("Hardware Information")
        report.append("-" * 40)
        report.append(f"GPU: {gpu_info.get('name', 'N/A')}")
        report.append(f"Memory: {gpu_info.get('total_memory_gb', 0):.2f} GB")
        report.append("")
        
        # Summary for each model
        report.append("Model Performance Summary (Batch Size = 1)")
        report.append("-" * 40)
        
        for model_name, model_results in results.items():
            report.append(f"\n{model_name}:")
            
            batch1_results = [r for r in model_results["benchmarks"] 
                            if r["batch_size"] == 1]
            
            if not batch1_results:
                continue
            
            avg_fps = np.mean([r["fps"] for r in batch1_results])
            avg_latency = np.mean([r["latency_ms"] for r in batch1_results])
            avg_memory = np.mean([r["gpu_memory_gb"] for r in batch1_results])
            
            realtime = "✓ YES" if avg_fps >= self.target_fps else "✗ NO"
            
            report.append(f"  Average FPS: {avg_fps:.2f}")
            report.append(f"  Average Latency: {avg_latency:.2f} ms")
            report.append(f"  GPU Memory: {avg_memory:.2f} GB")
            report.append(f"  Real-time capable (≥30 FPS): {realtime}")
        
        report.append("")
        
        # Real-time viability assessment
        report.append("Real-time Viability Assessment")
        report.append("-" * 40)
        
        realtime_models = []
        slow_models = []
        
        for model_name, model_results in results.items():
            batch1_results = [r for r in model_results["benchmarks"] 
                            if r["batch_size"] == 1]
            if batch1_results:
                avg_fps = np.mean([r["fps"] for r in batch1_results])
                if avg_fps >= self.target_fps:
                    realtime_models.append((model_name, avg_fps))
                else:
                    slow_models.append((model_name, avg_fps))
        
        if realtime_models:
            report.append("\n✓ Real-time capable models:")
            for name, fps in realtime_models:
                report.append(f"  • {name}: {fps:.2f} FPS")
        
        if slow_models:
            report.append("\n✗ Below real-time threshold:")
            for name, fps in slow_models:
                shortfall = self.target_fps - fps
                report.append(f"  • {name}: {fps:.2f} FPS ({shortfall:.2f} FPS short)")
        
        report.append("")
        
        # Recommendations
        report.append("Recommendations")
        report.append("-" * 40)
        
        if slow_models:
            report.append("• Consider model optimization techniques:")
            report.append("  - TensorRT optimization")
            report.append("  - Mixed precision (FP16)")
            report.append("  - Model pruning/quantization")
            report.append("  - Reduce input resolution")
        else:
            report.append("• All models meet real-time requirements")
            report.append("• Consider accuracy vs speed trade-offs")
        
        report.append("")
        report.append("="*70)
        
        return "\n".join(report)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark model FPS performance")
    parser.add_argument("--output", type=str, default="benchmarks",
                       help="Output directory for results")
    parser.add_argument("--models", type=str, nargs='+', required=True,
                       help="List of model config files")
    
    args = parser.parse_args()
    
    benchmark = FPSBenchmark(args.output)
    
    print("="*70)
    print("FPS Benchmarking for Real-time Perception")
    print("="*70)
    
    # Note: Actual model loading would happen here
    # This is a template - integrate with InternImage's model loading code
    
    print("\nNote: This script requires integration with model loading code.")
    print("Please adapt the model loading section for your specific setup.")

if __name__ == "__main__":
    main()
