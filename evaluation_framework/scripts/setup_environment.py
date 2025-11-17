"""
Environment Setup Script for InternImage DCNv3 Evaluation

This script sets up the complete environment for running InternImage evaluation
on Waymo and nuScenes datasets, including:
- Dependency installation
- DCNv3 operator compilation
- Dataset directory structure
- Pre-trained model downloads
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
import urllib.request
import shutil

class EnvironmentSetup:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.internimage_root = self.base_dir.parent
        self.detection_dir = self.internimage_root / "detection"
        self.auto_driving_dir = self.internimage_root / "autonomous_driving"
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("Checking Python version...")
        version = sys.version_info
        if version.major == 3 and version.minor >= 7:
            print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
            return True
        else:
            print(f"✗ Python {version.major}.{version.minor} is not compatible. Need Python 3.7+")
            return False
    
    def check_cuda(self):
        """Check CUDA availability"""
        print("\nChecking CUDA availability...")
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ CUDA is available")
                print(result.stdout)
                return True
        except FileNotFoundError:
            print("✗ CUDA (nvcc) not found. Please install CUDA toolkit.")
            return False
    
    def install_pytorch(self, cuda_version="11.3"):
        """Install PyTorch with CUDA support"""
        print(f"\nInstalling PyTorch with CUDA {cuda_version}...")
        
        # Map CUDA version to torch package
        cuda_map = {
            "11.3": "cu113",
            "11.7": "cu117",
            "11.8": "cu118",
            "12.1": "cu121"
        }
        
        cuda_suffix = cuda_map.get(cuda_version, "cu113")
        torch_version = "1.11.0" if cuda_version == "11.3" else "2.0.0"
        
        cmd = [
            sys.executable, "-m", "pip", "install",
            f"torch=={torch_version}+{cuda_suffix}",
            f"torchvision==0.12.0+{cuda_suffix}" if cuda_version == "11.3" else f"torchvision==0.15.0+{cuda_suffix}",
            "-f", "https://download.pytorch.org/whl/torch_stable.html"
        ]
        
        subprocess.run(cmd, check=True)
        print("✓ PyTorch installed successfully")
    
    def install_dependencies(self):
        """Install all required dependencies"""
        print("\nInstalling dependencies...")
        
        dependencies = [
            "opencv-python",
            "termcolor",
            "yacs",
            "pyyaml",
            "scipy",
            "timm==0.6.11",
            "numpy==1.26.4",
            "pydantic==1.10.13",
            "yapf==0.40.1",
            "openmim",
            "matplotlib",
            "seaborn",
            "pandas",
            "tqdm",
            "tensorboard",
            "pycocotools",
            "nuscenes-devkit",
            "waymo-open-dataset-tf-2-11-0"
        ]
        
        for dep in dependencies:
            print(f"  Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         check=True, stdout=subprocess.DEVNULL)
        
        print("✓ All dependencies installed")
    
    def install_mmcv(self):
        """Install mmcv-full and related packages"""
        print("\nInstalling mmcv-full...")
        
        subprocess.run([sys.executable, "-m", "mim", "install", "mmcv-full==1.5.0"], 
                      check=True)
        subprocess.run([sys.executable, "-m", "mim", "install", "mmsegmentation==0.27.0"], 
                      check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "mmdet==2.28.1"], 
                      check=True)
        
        print("✓ mmcv-full and mmdetection installed")
    
    def compile_dcnv3(self):
        """Compile DCNv3 operators"""
        print("\nCompiling DCNv3 operators...")
        
        dcnv3_dir = self.detection_dir / "ops_dcnv3"
        
        if not dcnv3_dir.exists():
            print(f"✗ DCNv3 directory not found at {dcnv3_dir}")
            return False
        
        original_dir = os.getcwd()
        os.chdir(dcnv3_dir)
        
        try:
            # Run make script
            if sys.platform == "win32":
                subprocess.run(["sh", "make.sh"], check=True, shell=True)
            else:
                subprocess.run(["bash", "make.sh"], check=True)
            
            # Test compilation
            print("  Testing DCNv3 compilation...")
            result = subprocess.run([sys.executable, "test.py"], 
                                  capture_output=True, text=True)
            
            if "All checks passed" in result.stdout or result.returncode == 0:
                print("✓ DCNv3 compiled successfully")
                return True
            else:
                print("✗ DCNv3 compilation test failed")
                print(result.stdout)
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"✗ DCNv3 compilation failed: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def create_data_directories(self):
        """Create directory structure for datasets"""
        print("\nCreating data directories...")
        
        data_dirs = [
            self.base_dir / "data" / "waymo",
            self.base_dir / "data" / "waymo" / "train",
            self.base_dir / "data" / "waymo" / "val",
            self.base_dir / "data" / "nuscenes",
            self.base_dir / "data" / "nuscenes" / "samples",
            self.base_dir / "data" / "nuscenes" / "v1.0-trainval",
            self.base_dir / "models" / "pretrained",
            self.base_dir / "results" / "waymo",
            self.base_dir / "results" / "nuscenes",
            self.base_dir / "results" / "baselines",
            self.base_dir / "results" / "cross_dataset",
        ]
        
        for dir_path in data_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {dir_path}")
        
        print("✓ Data directories created")
    
    def download_pretrained_models(self):
        """Download pre-trained InternImage models"""
        print("\nDownloading pre-trained models...")
        print("NOTE: This will download several GB of data. Please be patient.")
        
        models = {
            "mask_rcnn_internimage_t_fpn_1x_coco.pth": 
                "https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_1x_coco.pth",
            "mask_rcnn_internimage_s_fpn_1x_coco.pth": 
                "https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_s_fpn_1x_coco.pth",
            "mask_rcnn_internimage_b_fpn_1x_coco.pth": 
                "https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_b_fpn_1x_coco.pth",
        }
        
        model_dir = self.base_dir / "models" / "pretrained"
        
        for model_name, url in models.items():
            model_path = model_dir / model_name
            
            if model_path.exists():
                print(f"  {model_name} already exists, skipping...")
                continue
            
            print(f"  Downloading {model_name}...")
            try:
                # Note: For large files, consider using wget or curl
                print(f"    URL: {url}")
                print(f"    Please download manually to: {model_path}")
                print(f"    Or use: wget {url} -O {model_path}")
            except Exception as e:
                print(f"  ✗ Failed to download {model_name}: {e}")
        
        print("✓ Model download instructions provided")
    
    def create_config_template(self):
        """Create configuration templates"""
        print("\nCreating configuration templates...")
        
        # Create experiment config
        config = {
            "experiment_name": "dcnv3_evaluation",
            "datasets": {
                "waymo": {
                    "data_root": str(self.base_dir / "data" / "waymo"),
                    "split": "val",
                    "num_samples": 1000
                },
                "nuscenes": {
                    "data_root": str(self.base_dir / "data" / "nuscenes"),
                    "version": "v1.0-trainval",
                    "split": "val"
                }
            },
            "models": {
                "internimage_t": {
                    "config": "configs/coco/mask_rcnn_internimage_t_fpn_1x_coco.py",
                    "checkpoint": str(self.base_dir / "models" / "pretrained" / "mask_rcnn_internimage_t_fpn_1x_coco.pth")
                },
                "internimage_s": {
                    "config": "configs/coco/mask_rcnn_internimage_s_fpn_1x_coco.py",
                    "checkpoint": str(self.base_dir / "models" / "pretrained" / "mask_rcnn_internimage_s_fpn_1x_coco.pth")
                }
            },
            "evaluation": {
                "batch_size": 1,
                "gpu_id": 0,
                "distance_bins": [0, 30, 50, 100],
                "target_fps": 30,
                "metrics": ["mAP", "AP50", "AP75", "pedestrian_recall"]
            },
            "output_dir": str(self.base_dir / "results")
        }
        
        config_path = self.base_dir / "configs" / "base_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, indent=2, fp=f)
        
        print(f"✓ Configuration template created at {config_path}")
    
    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "="*70)
        print("Environment Setup Complete!")
        print("="*70)
        print("\n📋 Next Steps:\n")
        print("1. Download datasets:")
        print("   - Waymo Open Dataset: https://waymo.com/open/")
        print("   - nuScenes: https://www.nuscenes.org/")
        print(f"   - Place in: {self.base_dir / 'data'}")
        print()
        print("2. Download pre-trained models (see URLs above)")
        print(f"   - Place in: {self.base_dir / 'models' / 'pretrained'}")
        print()
        print("3. Verify installation:")
        print("   python -c 'import torch; print(torch.cuda.is_available())'")
        print()
        print("4. Run test evaluation:")
        print("   python scripts/run_waymo_eval.py --config configs/base_config.json")
        print()
        print("📚 Documentation:")
        print("   - Framework README: evaluation_framework/README.md")
        print("   - InternImage docs: README.md")
        print()
        print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Setup InternImage evaluation environment")
    parser.add_argument("--base-dir", type=str, 
                       default=None,
                       help="Base directory for evaluation framework")
    parser.add_argument("--cuda-version", type=str, default="11.3",
                       help="CUDA version (11.3, 11.7, 11.8, 12.1)")
    parser.add_argument("--skip-pytorch", action="store_true",
                       help="Skip PyTorch installation")
    parser.add_argument("--skip-dcnv3", action="store_true",
                       help="Skip DCNv3 compilation")
    parser.add_argument("--skip-models", action="store_true",
                       help="Skip model download")
    
    args = parser.parse_args()
    
    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        # Assume script is in evaluation_framework/scripts/
        base_dir = Path(__file__).parent.parent
    
    print("="*70)
    print("InternImage DCNv3 Evaluation Environment Setup")
    print("="*70)
    print(f"\nBase directory: {base_dir}")
    
    setup = EnvironmentSetup(base_dir)
    
    # Check prerequisites
    if not setup.check_python_version():
        sys.exit(1)
    
    if not setup.check_cuda():
        print("\nWarning: CUDA not found. GPU acceleration will not be available.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Install PyTorch
    if not args.skip_pytorch:
        try:
            setup.install_pytorch(args.cuda_version)
        except Exception as e:
            print(f"✗ PyTorch installation failed: {e}")
            sys.exit(1)
    
    # Install dependencies
    try:
        setup.install_dependencies()
        setup.install_mmcv()
    except Exception as e:
        print(f"✗ Dependency installation failed: {e}")
        sys.exit(1)
    
    # Compile DCNv3
    if not args.skip_dcnv3:
        if not setup.compile_dcnv3():
            print("\n⚠ DCNv3 compilation failed. You may need to install manually.")
            print("   See: detection/ops_dcnv3/README.md")
    
    # Setup directories and configs
    setup.create_data_directories()
    setup.create_config_template()
    
    # Download models
    if not args.skip_models:
        setup.download_pretrained_models()
    
    # Print next steps
    setup.print_next_steps()

if __name__ == "__main__":
    main()
