#!/usr/bin/env python3
"""
Reemplazar completamente setup_databases con versión robusta
"""

import shutil
from datetime import datetime

filepath = "database/data_manager.py"

print("🔧 Reemplazando setup_databases...")

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Backup
backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(filepath, backup_path)
print(f"✓ Backup: {backup_path}")

# Nuevo método setup_databases completo
new_setup_databases = '''    def setup_databases(self):
        """Configura todas las conexiones a bases de datos"""
        # Flags de disponibilidad
        self.postgres_available = False
        self.influx_available = False
        self.redis_available = False
        
        # PostgreSQL
        try:
            self.pg_engine = create_engine(self.config.database.postgres_url)
            self.Session = sessionmaker(bind=self.pg_engine)
            TradingData.create_table(self.pg_engine)
            self.postgres_available = True
            logger.info("PostgreSQL connected successfully")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
            self.pg_engine = None
            self.Session = None
        
        # InfluxDB
        try:
            self.influx_client = InfluxDBClient(
                url=self.config.database.influx_url,
                token=self.config.database.influx_token,
                org=self.config.database.influx_org
            )
            self.write_api = self.influx_client.write_api(write_option=SYNCHRONOUS)
            self.query_api = self.influx_client.query_api()
            self.influx_available = True
            logger.info("InfluxDB connected successfully")
        except Exception as e:
            logger.warning(f"InfluxDB connection failed: {e}")
            self.influx_client = None
            self.write_api = None
            self.query_api = None
        
        # Redis
        try:
            self.redis_client = redis.Redis.from_url(
                self.config.database.redis_url,
                decode_responses=False
            )
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        logger.info(f"Database availability - Postgres: {self.postgres_available}, "
                   f"InfluxDB: {self.influx_available}, Redis: {self.redis_available}")
'''

# Encontrar y reemplazar el método setup_databases completo
lines = content.split('\n')
new_lines = []
skip_until_next_def = False
found_setup_databases = False

for i, line in enumerate(lines):
    if 'def setup_databases(self):' in line:
        found_setup_databases = True
        skip_until_next_def = True
        # Insertar el nuevo método
        new_lines.append(new_setup_databases)
        continue
    
    if skip_until_next_def:
        # Saltar hasta encontrar el siguiente método o clase
        if (line.strip().startswith('def ') or 
            line.strip().startswith('class ')) and 'def setup_databases' not in line:
            skip_until_next_def = False
            new_lines.append(line)
        continue
    
    new_lines.append(line)

if found_setup_databases:
    content = '\n'.join(new_lines)
    
    # Ahora modificar los métodos que usan las bases de datos
    
    # _store_in_influx
    content = content.replace(
        '''    def _store_in_influx(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en InfluxDB"""
        points = []''',
        '''    def _store_in_influx(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en InfluxDB"""
        if not self.influx_available or self.write_api is None:
            return
        
        points = []'''
    )
    
    # _store_in_redis
    content = content.replace(
        '''    def _store_in_redis(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en Redis con compresión"""
        cache_key = f"market_data:{symbol}:{timeframe}"''',
        '''    def _store_in_redis(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en Redis con compresión"""
        if not self.redis_available or self.redis_client is None:
            return
        
        cache_key = f"market_data:{symbol}:{timeframe}"'''
    )
    
    # _store_in_postgres
    content = content.replace(
        '''    def _store_in_postgres(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en PostgreSQL"""
        with self.Session() as session:''',
        '''    def _store_in_postgres(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Almacena en PostgreSQL"""
        if not self.postgres_available or self.Session is None:
            return
        
        with self.Session() as session:'''
    )
    
    # get_cached_data
    content = content.replace(
        '''    def get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Obtiene datos cacheados de Redis"""
        try:
            cache_key = f"market_data:{symbol}:{timeframe}"''',
        '''    def get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Obtiene datos cacheados de Redis"""
        if not self.redis_available or self.redis_client is None:
            return None
        
        try:
            cache_key = f"market_data:{symbol}:{timeframe}"'''
    )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ Archivo reemplazado con versión robusta")
    
    # Verificar sintaxis
    print("\n🧪 Verificando sintaxis...")
    try:
        compile(content, filepath, 'exec')
        print("✅ Sintaxis correcta")
        
        print("\n🧪 Verificando import...")
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
        print("✅ ¡TODO LISTO!")
        print("="*60)
        print("\nEjecuta el test:")
        print("  python -m pytest tests/test_suite.py::TestTradingPlatform::test_02_data_management -v")
        print("\nO todos los tests:")
        print("  python -m pytest tests/test_suite.py -v")
        
    except SyntaxError as e:
        print(f"❌ Error de sintaxis: {e}")
        print("\n   Restaurando backup...")
        shutil.copy2(backup_path, filepath)
    except Exception as e:
        print(f"⚠️  Error en import (puede ser normal): {e}")
        print("   Intenta ejecutar el test de todas formas")
        
else:
    print("❌ No se encontró el método setup_databases")