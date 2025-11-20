# test_installation.py
try:
    import pandas as pd
    import numpy as np
    import sklearn
    import tensorflow as tf
    import MetaTrader5 as mt5
    import yfinance as yf
    print("✅ Todas las dependencias críticas funcionan!")
    
    # Probar si tenemos alternativas a TA-Lib
    try:
        import talib
        print("✅ TA-Lib disponible")
    except ImportError:
        try:
            import pandas_ta
            print("✅ pandas_ta disponible como alternativa")
        except ImportError:
            print("⚠️  Indicadores técnicos limitados - instala TA-Lib o pandas_ta")
            
except ImportError as e:
    print(f"❌ Error: {e}")