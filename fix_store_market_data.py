#!/usr/bin/env python3
"""
Asegurar que store_market_data siempre almacene en cache (memoria o Redis)
"""

import shutil
from datetime import datetime

filepath = "database/data_manager.py"

print("🔧 Arreglando store_market_data...")

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Backup
backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(filepath, backup_path)
print(f"✓ Backup: {backup_path}")

# Buscar store_market_data y ver su contenido
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'def store_market_data(' in line:
        print(f"\n📍 Encontrado store_market_data en línea {i+1}")
        print("   Contenido:")
        for j in range(i, min(i+30, len(lines))):
            print(f"   {j+1}: {lines[j]}")
        break

# Reemplazar store_market_data para que siempre use cache
new_store_market_data = '''    def store_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena datos de mercado en múltiples bases de datos"""
        success = False
        
        try:
            # PostgreSQL para consultas estructuradas
            self._store_in_postgres(symbol, timeframe, data)
            if self.postgres_available:
                success = True
        except Exception as e:
            logger.error(f"Error storing in PostgreSQL: {e}")
        
        try:
            # InfluxDB para análisis temporal
            self._store_in_influx(symbol, timeframe, data)
            if self.influx_available:
                success = True
        except Exception as e:
            logger.error(f"Error storing in InfluxDB: {e}")
        
        try:
            # Redis/Memory para cache (SIEMPRE se ejecuta)
            self._store_in_redis(symbol, timeframe, data)
            success = True  # Cache siempre funciona (Redis o memoria)
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
        
        if not success:
            logger.warning("Data was not stored in any database")
'''

# Encontrar y reemplazar store_market_data
found = False
new_lines = []
skip_until_next_def = False

for i, line in enumerate(lines):
    if 'def store_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame):' in line:
        found = True
        skip_until_next_def = True
        new_lines.append(new_store_market_data)
        continue
    
    if skip_until_next_def:
        # Saltar hasta el siguiente método
        if (line.strip().startswith('def ') or 
            line.strip().startswith('class ')) and 'def store_market_data' not in line:
            skip_until_next_def = False
            new_lines.append(line)
        continue
    
    new_lines.append(line)

if found:
    content = '\n'.join(new_lines)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ store_market_data actualizado")
    print("   Ahora SIEMPRE usa cache (Redis o memoria)")
    
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
        print("\nEjecuta:")
        print("  python -m pytest tests/test_suite.py::TestTradingPlatform::test_02_data_management -v")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
else:
    print("\n❌ No se encontró store_market_data")