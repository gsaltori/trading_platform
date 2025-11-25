#!/usr/bin/env python3
"""
Agregar cache en memoria como fallback cuando Redis no está disponible
"""

import shutil
from datetime import datetime

filepath = "database/data_manager.py"

print("🔧 Agregando cache en memoria como fallback...")

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Backup
backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(filepath, backup_path)
print(f"✓ Backup: {backup_path}")

# Agregar cache en memoria en __init__
if 'self.memory_cache = {}' not in content:
    content = content.replace(
        'def __init__(self, config):',
        '''def __init__(self, config):
        self.memory_cache = {}  # Cache en memoria como fallback'''
    )
    print("   ✓ Agregado memory_cache en __init__")

# Modificar _store_in_redis para usar memoria si Redis no disponible
content = content.replace(
    '''    def _store_in_redis(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en Redis con compresión"""
        if not self.redis_available or self.redis_client is None:
            return
        
        cache_key = f"market_data:{symbol}:{timeframe}"
        compressed_data = zlib.compress(pickle.dumps(data))
        self.redis_client.setex(cache_key, 3600, compressed_data)  # 1 hora''',
    '''    def _store_in_redis(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en Redis con compresión, o en memoria si Redis no disponible"""
        cache_key = f"market_data:{symbol}:{timeframe}"
        
        if self.redis_available and self.redis_client is not None:
            # Usar Redis si está disponible
            compressed_data = zlib.compress(pickle.dumps(data))
            self.redis_client.setex(cache_key, 3600, compressed_data)  # 1 hora
        else:
            # Fallback a cache en memoria
            self.memory_cache[cache_key] = data.copy()
            logger.debug(f"Stored {cache_key} in memory cache (Redis not available)")'''
)

# Modificar get_cached_data para usar memoria si Redis no disponible
content = content.replace(
    '''    def get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Obtiene datos cacheados de Redis"""
        if not self.redis_available or self.redis_client is None:
            return None
        
        try:
            cache_key = f"market_data:{symbol}:{timeframe}"
            cached = self.redis_client.get(cache_key)
            if cached:
                return pickle.loads(zlib.decompress(cached))
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
        return None''',
    '''    def get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Obtiene datos cacheados de Redis o memoria"""
        cache_key = f"market_data:{symbol}:{timeframe}"
        
        # Intentar Redis primero
        if self.redis_available and self.redis_client is not None:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return pickle.loads(zlib.decompress(cached))
            except Exception as e:
                logger.error(f"Error getting cached data from Redis: {e}")
        
        # Fallback a memoria
        if cache_key in self.memory_cache:
            logger.debug(f"Retrieved {cache_key} from memory cache")
            return self.memory_cache[cache_key].copy()
        
        return None'''
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ Cache en memoria agregado como fallback")

# Verificar
print("\n🧪 Verificando...")
try:
    import sys
    for module in list(sys.modules.keys()):
        if 'database' in module or 'core' in module:
            try:
                del sys.modules[module]
            except:
                pass
    
    from database.data_manager import DataManager
    print("✅ Import exitoso")
    
    print("\n" + "="*60)
    print("✅ ¡LISTO!")
    print("="*60)
    print("\nAhora el DataManager:")
    print("  ✓ Usa Redis si está disponible")
    print("  ✓ Usa memoria si Redis no está disponible")
    print("  ✓ El test debería pasar")
    print("\nEjecuta:")
    print("  python -m pytest tests/test_suite.py -v")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n   Restaurando backup...")
    shutil.copy2(backup_path, filepath)