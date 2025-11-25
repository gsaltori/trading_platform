#!/usr/bin/env python3
"""
Test rápido para ver qué está pasando
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Limpiar módulos
for module in list(sys.modules.keys()):
    if 'database' in module or 'core' in module or 'config' in module:
        try:
            del sys.modules[module]
        except:
            pass

print("🧪 Test rápido de DataManager...")

try:
    from core.platform import TradingPlatform
    
    # Crear datos de prueba
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    print("✓ Datos de prueba creados")
    
    # Crear plataforma
    platform = TradingPlatform()
    platform.initialize()
    
    print(f"✓ Plataforma inicializada")
    print(f"   memory_cache exists: {hasattr(platform.data_manager, 'memory_cache')}")
    print(f"   redis_available: {platform.data_manager.redis_available}")
    
    # Almacenar datos
    print("\n📝 Almacenando datos...")
    platform.data_manager.store_market_data("TEST", "D1", test_data)
    
    # Verificar cache en memoria
    cache_key = "market_data:TEST:D1"
    print(f"\n🔍 Verificando cache...")
    print(f"   Claves en memory_cache: {list(platform.data_manager.memory_cache.keys())}")
    print(f"   Cache key buscada: {cache_key}")
    print(f"   ¿Está en cache?: {cache_key in platform.data_manager.memory_cache}")
    
    # Intentar recuperar
    print("\n📥 Recuperando datos...")
    cached_data = platform.data_manager.get_cached_data("TEST", "D1")
    
    if cached_data is not None:
        print(f"✅ ¡ÉXITO! Datos recuperados: {len(cached_data)} filas")
    else:
        print(f"❌ FALLÓ: cached_data es None")
        
        # Debug adicional
        print("\n🔍 Debug adicional:")
        print(f"   redis_available: {platform.data_manager.redis_available}")
        print(f"   redis_client: {platform.data_manager.redis_client}")
        print(f"   memory_cache: {platform.data_manager.memory_cache}")
    
    platform.shutdown()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()