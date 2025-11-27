#!/usr/bin/env python3
# run_gui.py
"""
Quick launcher for the Trading Platform GUI.

Usage:
    python run_gui.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Launch the GUI."""
    print("=" * 60)
    print("   TRADING PLATFORM - Strategy Generator")
    print("=" * 60)
    print()
    
    # Check dependencies
    missing_deps = []
    
    try:
        import pandas
        print(f"  ✓ pandas {pandas.__version__}")
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import numpy
        print(f"  ✓ numpy {numpy.__version__}")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import sklearn
        print(f"  ✓ scikit-learn {sklearn.__version__}")
    except ImportError:
        missing_deps.append("scikit-learn")
    
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QT_VERSION_STR
        print(f"  ✓ PyQt6 {QT_VERSION_STR}")
    except ImportError:
        missing_deps.append("PyQt6")
    
    try:
        import matplotlib
        print(f"  ✓ matplotlib {matplotlib.__version__}")
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import ta
        print(f"  ✓ ta (technical analysis)")
    except ImportError:
        print("  ⚠ ta (optional, will use basic indicators)")
    
    try:
        import MetaTrader5
        print(f"  ✓ MetaTrader5")
    except ImportError:
        print("  ⚠ MetaTrader5 (optional, for live trading)")
    
    print()
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print()
        print("Install with:")
        print(f"   pip install {' '.join(missing_deps)}")
        return 1
    
    print("Starting GUI...")
    print("=" * 60)
    print()
    
    try:
        from ui.main_window import run_gui
        return run_gui()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
