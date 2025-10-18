#!/usr/bin/env python3
"""
Quick verification script to check if environment is ready for USDJPY training
"""

import sys

def check_imports():
    """Check if all required packages are installed"""
    print("Checking required packages...\n")
    
    packages = {
        'torch': 'PyTorch',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'pandas_ta': 'Technical Analysis library'
    }
    
    missing = []
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages installed!")
        return True


def check_project_structure():
    """Check if required directories and files exist"""
    print("\n" + "="*60)
    print("Checking project structure...\n")
    
    import os
    
    required = {
        'src/tcn_model.py': 'TCN Model',
        'src/data_preprocessing.py': 'Data Preprocessor',
        'utils/logging_utils.py': 'Logging utilities',
        'utils/backtesting.py': 'Backtesting engine',
        'train_usdjpy.py': 'USDJPY training script'
    }
    
    missing = []
    for path, desc in required.items():
        if os.path.exists(path):
            print(f"✅ {desc} ({path})")
        else:
            print(f"❌ {desc} ({path}) - MISSING")
            missing.append(path)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False
    else:
        print("\n✅ All required files present!")
        return True


def check_cuda():
    """Check CUDA availability for GPU acceleration"""
    print("\n" + "="*60)
    print("Checking GPU availability...\n")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available!")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Training will be FAST 🚀")
        else:
            print("⚠️  CUDA not available - will use CPU")
            print("   Training will be slower but still works")
    except:
        print("❌ Could not check CUDA")


def provide_instructions():
    """Provide next steps"""
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. Place your USDJPY CSV file in the data/ directory:
   cp /path/to/your/usdjpy.csv data/usdjpy_100k.csv

2. Verify your CSV has these columns (any variation):
   - timestamp/date/time
   - open/o
   - high/h
   - low/l
   - close/c
   - volume/v (optional)

3. Run training:
   python train_usdjpy.py --data_path data/usdjpy_100k.csv

4. Monitor progress:
   - Training will take 5-15 minutes on GPU
   - Training will take 20-45 minutes on CPU
   - Watch for validation accuracy to stabilize

5. Check results:
   - Model: models/tcn_usdjpy.pt
   - Backtest: results/usdjpy_backtest.png
   - Logs: logs/training_*.log

For detailed instructions, see: USDJPY_TRAINING_GUIDE.md
""")


def main():
    print("="*60)
    print("TCN USDJPY Training - Environment Check")
    print("="*60)
    
    checks_passed = 0
    total_checks = 2
    
    if check_imports():
        checks_passed += 1
    
    if check_project_structure():
        checks_passed += 1
    
    check_cuda()
    
    print("\n" + "="*60)
    if checks_passed == total_checks:
        print("✅ ENVIRONMENT READY!")
        print("="*60)
        provide_instructions()
        return 0
    else:
        print("❌ ENVIRONMENT NOT READY")
        print(f"   Passed {checks_passed}/{total_checks} checks")
        print("="*60)
        print("\nPlease fix the issues above before training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
