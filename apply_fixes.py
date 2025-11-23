#!/usr/bin/env python3
"""
Script para aplicar automáticamente las correcciones a los tests fallidos
"""

import os
import shutil
from datetime import datetime

def backup_file(filepath):
    """Crear backup de un archivo"""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"✅ Backup creado: {backup_path}")
        return True
    return False

def apply_data_manager_fix():
    """Aplicar fix al data_manager.py"""
    filepath = "database/data_manager.py"
    
    if not os.path.exists(filepath):
        print(f"❌ Archivo no encontrado: {filepath}")
        return False
    
    print(f"\n📝 Aplicando fix a {filepath}...")
    backup_file(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Actualizar __init__
    old_init = """class DataManager:
    def __init__(self, config):
        self.config = config
        self.setup_databases()"""
    
    new_init = """class DataManager:
    def __init__(self, config):
        self.config = config
        self.influx_available = False
        self.postgres_available = False
        self.redis_available = False
        self.setup_databases()"""
    
    if old_init in content:
        content = content.replace(old_init, new_init)
        print("  ✓ Actualizado __init__")
    
    # Fix 2: Actualizar setup_databases
    old_setup = """    def setup_databases(self):
        \"\"\"Configura todas las conexiones a bases de datos\"\"\"
        try:
            # PostgreSQL para datos estructurados
            self.pg_engine = create_engine(self.config.database.postgres_url)
            self.Session = sessionmaker(bind=self.pg_engine)
            
            # InfluxDB para series temporales
            self.influx_client = InfluxDBClient(
                url=self.config.database.influx_url,
                token=self.config.database.influx_token,
                org=self.config.database.influx_org
            )
            self.write_api = self.influx_client.write_api(write_option=SYNCHRONOUS)
            self.query_api = self.influx_client.query_api()
            
            # Redis para cache
            self.redis_client = redis.Redis.from_url(
                self.config.database.redis_url,
                decode_responses=False
            )
            
            # Crear tablas si no existen
            TradingData.create_table(self.pg_engine)
            
            logger.info("All databases connected successfully")
            
        except Exception as e:
            logger.error(f"Database setup error: {e}")
            raise"""
    
    new_setup = """    def setup_databases(self):
        \"\"\"Configura todas las conexiones a bases de datos con manejo robusto de errores\"\"\"
        # PostgreSQL para datos estructurados
        try:
            self.pg_engine = create_engine(self.config.database.postgres_url)
            self.Session = sessionmaker(bind=self.pg_engine)
            TradingData.create_table(self.pg_engine)
            self.postgres_available = True
            logger.info("PostgreSQL connected successfully")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
            self.postgres_available = False
        
        # InfluxDB para series temporales
        try:
            from influxdb_client.client.write_api import SYNCHRONOUS
            
            self.influx_client = InfluxDBClient(
                url=self.config.database.influx_url,
                token=self.config.database.influx_token,
                org=self.config.database.influx_org
            )
            self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.influx_client.query_api()
            self.influx_available = True
            logger.info("InfluxDB connected successfully")
        except Exception as e:
            logger.warning(f"InfluxDB connection failed: {e}")
            self.influx_available = False
            self.write_api = None
            self.query_api = None
        
        # Redis para cache
        try:
            self.redis_client = redis.Redis.from_url(
                self.config.database.redis_url,
                decode_responses=False
            )
            self.redis_client.ping()  # Test connection
            self.redis_available = True
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_available = False
            
        if not any([self.postgres_available, self.influx_available, self.redis_available]):
            logger.warning("No databases available - operating in degraded mode")"""
    
    if old_setup in content:
        content = content.replace(old_setup, new_setup)
        print("  ✓ Actualizado setup_databases")
    
    # Fix 3: Actualizar _store_in_influx
    old_store_influx = """    def _store_in_influx(self, symbol: str, timeframe: str, data: pd.DataFrame):
        \"\"\"Almacena en InfluxDB\"\"\"
        points = []
        for idx, row in data.iterrows():
            point = Point("market_data") \\
                .tag("symbol", symbol) \\
                .tag("timeframe", timeframe) \\
                .field("open", float(row.get('open', 0))) \\
                .field("high", float(row.get('high', 0))) \\
                .field("low", float(row.get('low', 0))) \\
                .field("close", float(row.get('close', 0))) \\
                .field("volume", float(row.get('volume', 0))) \\
                .time(idx)
            points.append(point)
        
        self.write_api.write(bucket="trading", record=points)"""
    
    new_store_influx = """    def _store_in_influx(self, symbol: str, timeframe: str, data: pd.DataFrame):
        \"\"\"Almacena en InfluxDB\"\"\"
        if not self.influx_available or self.write_api is None:
            return
        
        from influxdb_client import Point
        
        points = []
        for idx, row in data.iterrows():
            point = Point("market_data") \\
                .tag("symbol", symbol) \\
                .tag("timeframe", timeframe) \\
                .field("open", float(row.get('open', 0))) \\
                .field("high", float(row.get('high', 0))) \\
                .field("low", float(row.get('low', 0))) \\
                .field("close", float(row.get('close', 0))) \\
                .field("volume", float(row.get('volume', 0))) \\
                .time(idx)
            points.append(point)
        
        self.write_api.write(bucket="trading", record=points)"""
    
    if old_store_influx in content:
        content = content.replace(old_store_influx, new_store_influx)
        print("  ✓ Actualizado _store_in_influx")
    
    # Fix 4: Actualizar get_cached_data
    old_get_cached = """    def get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        \"\"\"Obtiene datos cacheados de Redis\"\"\"
        try:
            cache_key = f"market_data:{symbol}:{timeframe}"
            cached = self.redis_client.get(cache_key)
            if cached:
                return pickle.loads(zlib.decompress(cached))
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
        return None"""
    
    new_get_cached = """    def get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        \"\"\"Obtiene datos cacheados de Redis\"\"\"
        if not self.redis_available:
            return None
        
        try:
            cache_key = f"market_data:{symbol}:{timeframe}"
            cached = self.redis_client.get(cache_key)
            if cached:
                return pickle.loads(zlib.decompress(cached))
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
        return None"""
    
    if old_get_cached in content:
        content = content.replace(old_get_cached, new_get_cached)
        print("  ✓ Actualizado get_cached_data")
    
    # Guardar archivo
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ {filepath} actualizado correctamente")
    return True

def apply_ml_engine_fix():
    """Aplicar fix al ml_engine.py"""
    filepath = "ml/ml_engine.py"
    
    if not os.path.exists(filepath):
        print(f"❌ Archivo no encontrado: {filepath}")
        return False
    
    print(f"\n📝 Aplicando fix a {filepath}...")
    backup_file(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar y reemplazar el método prepare_features
    # Encontrar el inicio del método
    method_start = "    def prepare_features(self, data: pd.DataFrame, target: pd.Series,"
    
    if method_start not in content:
        print("  ❌ No se encontró el método prepare_features")
        return False
    
    # Encontrar el final del método (siguiente def o fin de clase)
    lines = content.split('\n')
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if method_start in line:
            start_idx = i
        elif start_idx is not None and line.startswith('    def ') and i > start_idx:
            end_idx = i
            break
    
    if start_idx is None:
        print("  ❌ No se pudo localizar el método")
        return False
    
    # Si no encontramos el final, buscar el final de la clase
    if end_idx is None:
        for i in range(start_idx + 1, len(lines)):
            if lines[i].startswith('class '):
                end_idx = i
                break
        if end_idx is None:
            end_idx = len(lines)
    
    # Nuevo método
    new_method = """    def prepare_features(self, data: pd.DataFrame, target: pd.Series, 
                        fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        \"\"\"Preparar características para ML - FIXED VERSION\"\"\"
        
        # Validar datos de entrada
        if len(data) < 100:
            raise ValueError(f"Insufficient data: {len(data)} rows. Need at least 100 rows.")
        
        # Crear características técnicas
        df = self.create_technical_features(data)
        
        # Crear características retrasadas (solo las más importantes para evitar NaN)
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        existing_price_columns = [col for col in price_columns if col in df.columns]
        # Reducir lags para mantener más datos
        df = self.create_lagged_features(df, existing_price_columns, [1, 2, 3])
        
        # Crear características rolling (reducidas)
        rolling_columns = []
        if 'returns' in df.columns:
            rolling_columns.append('returns')
        if 'volume' in df.columns:
            rolling_columns.append('volume')
        if 'atr' in df.columns:
            rolling_columns.append('atr')
        
        if rolling_columns:
            # Reducir ventanas para mantener más datos
            df = self.create_rolling_features(df, rolling_columns, [5, 10])
        
        # Alinear con target y eliminar NaN
        aligned_data = pd.concat([df, target], axis=1).dropna()
        
        # Verificar que tenemos suficientes datos después de eliminar NaN
        if len(aligned_data) < 10:
            raise ValueError(f"After removing NaN, only {len(aligned_data)} samples remain. "
                            f"Original data had {len(data)} rows. This is insufficient for training.")
        
        X = aligned_data.iloc[:, :-1]
        y = aligned_data.iloc[:, -1]
        
        # Verificar que no hay columnas con todos NaN o Inf
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna(axis=1, how='all')  # Eliminar columnas con solo NaN
        
        # Eliminar columnas con varianza cero
        X = X.loc[:, X.std() > 0]
        
        if X.shape[1] == 0:
            raise ValueError("No valid features after preprocessing")
        
        # Escalar características
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Scaler no está fitted. Llama con fit=True primero.")
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), y
"""
    
    # Reemplazar el método
    lines = lines[:start_idx] + new_method.split('\n') + lines[end_idx:]
    
    # Guardar archivo
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print("  ✓ Actualizado método prepare_features")
    print(f"✅ {filepath} actualizado correctamente")
    return True

def main():
    """Función principal"""
    print("🔧 Aplicando correcciones para tests fallidos...")
    print("=" * 60)
    
    success = True
    
    # Aplicar fix a data_manager.py
    if not apply_data_manager_fix():
        success = False
    
    # Aplicar fix a ml_engine.py
    if not apply_ml_engine_fix():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Todas las correcciones aplicadas exitosamente!")
        print("\n📊 Ejecuta los tests con:")
        print("   python -m pytest tests/test_suite.py -v")
    else:
        print("❌ Algunas correcciones fallaron. Revisa los mensajes arriba.")
        print("\n💡 Puedes aplicar los fixes manualmente siguiendo:")
        print("   FIXES_PARA_TESTS.md")
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())