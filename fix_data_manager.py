#!/usr/bin/env python3
"""
Arreglar database/data_manager.py para manejar servicios no disponibles
"""

import shutil
from datetime import datetime

filepath = "database/data_manager.py"

print("🔧 Aplicando fix de degradación elegante para bases de datos...")

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Backup
backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(filepath, backup_path)
print(f"✓ Backup: {backup_path}")

# Buscar setup_databases y agregar flags de disponibilidad
modified = False
new_lines = []
in_setup_databases = False
added_flags = False

for i, line in enumerate(lines):
    # Detectar inicio de setup_databases
    if 'def setup_databases(self):' in line:
        in_setup_databases = True
        new_lines.append(line)
        continue
    
    # Agregar flags de disponibilidad al inicio del método
    if in_setup_databases and not added_flags and 'try:' in line:
        # Agregar flags antes del try
        indent = ' ' * 8
        new_lines.append(f'{indent}# Flags de disponibilidad\n')
        new_lines.append(f'{indent}self.postgres_available = False\n')
        new_lines.append(f'{indent}self.influx_available = False\n')
        new_lines.append(f'{indent}self.redis_available = False\n')
        new_lines.append(f'{indent}\n')
        added_flags = True
        modified = True
    
    # Modificar el manejo de errores en setup_databases
    if in_setup_databases and 'except Exception as e:' in line:
        # Agregar manejo individual de cada servicio
        indent = ' ' * 8
        new_lines.append(f'{indent}# Intentar cada servicio individualmente\n')
        new_lines.append(f'{indent}try:\n')
        new_lines.append(f'{indent}    self.pg_engine = create_engine(self.config.database.postgres_url)\n')
        new_lines.append(f'{indent}    self.Session = sessionmaker(bind=self.pg_engine)\n')
        new_lines.append(f'{indent}    self.postgres_available = True\n')
        new_lines.append(f'{indent}except Exception as e:\n')
        new_lines.append(f'{indent}    logger.warning(f"PostgreSQL not available: {{e}}")\n')
        new_lines.append(f'{indent}\n')
        new_lines.append(f'{indent}try:\n')
        new_lines.append(f'{indent}    self.influx_client = InfluxDBClient(\n')
        new_lines.append(f'{indent}        url=self.config.database.influx_url,\n')
        new_lines.append(f'{indent}        token=self.config.database.influx_token,\n')
        new_lines.append(f'{indent}        org=self.config.database.influx_org\n')
        new_lines.append(f'{indent}    )\n')
        new_lines.append(f'{indent}    self.write_api = self.influx_client.write_api(write_option=SYNCHRONOUS)\n')
        new_lines.append(f'{indent}    self.query_api = self.influx_client.query_api()\n')
        new_lines.append(f'{indent}    self.influx_available = True\n')
        new_lines.append(f'{indent}except Exception as e:\n')
        new_lines.append(f'{indent}    logger.warning(f"InfluxDB not available: {{e}}")\n')
        new_lines.append(f'{indent}\n')
        new_lines.append(f'{indent}try:\n')
        new_lines.append(f'{indent}    self.redis_client = redis.Redis.from_url(\n')
        new_lines.append(f'{indent}        self.config.database.redis_url,\n')
        new_lines.append(f'{indent}        decode_responses=False\n')
        new_lines.append(f'{indent}    )\n')
        new_lines.append(f'{indent}    self.redis_client.ping()\n')
        new_lines.append(f'{indent}    self.redis_available = True\n')
        new_lines.append(f'{indent}except Exception as e:\n')
        new_lines.append(f'{indent}    logger.warning(f"Redis not available: {{e}}")\n')
        new_lines.append(f'{indent}\n')
        new_lines.append(f'{indent}logger.info(f"Databases: Postgres={{self.postgres_available}}, InfluxDB={{self.influx_available}}, Redis={{self.redis_available}}")\n')
        
        modified = True
        # Saltar las líneas originales del except
        in_setup_databases = False
        continue
    
    new_lines.append(line)

# Ahora modificar store_market_data para usar los flags
content = ''.join(new_lines)

# Modificar _store_in_influx
content = content.replace(
    'def _store_in_influx(self, symbol: str, timeframe: str, data: pd.DataFrame):',
    '''def _store_in_influx(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en InfluxDB"""
        if not self.influx_available:
            return'''
)

# Modificar _store_in_redis
content = content.replace(
    'def _store_in_redis(self, symbol: str, timeframe: str, data: pd.DataFrame):',
    '''def _store_in_redis(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en Redis con compresión"""
        if not self.redis_available:
            return'''
)

# Modificar get_cached_data
content = content.replace(
    'def get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:',
    '''def get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Obtiene datos cacheados de Redis"""
        if not self.redis_available:
            return None'''
)

if modified:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ Archivo modificado con degradación elegante")
    print("\nEjecuta el test:")
    print("  python -m pytest tests/test_suite.py::TestTradingPlatform::test_02_data_management -v")
else:
    print("\n⚠️  No se encontraron las líneas esperadas")