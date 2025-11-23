#!/usr/bin/env python3
"""
setup_lite_mode.py - Configuración automática de modo LITE
Configura la plataforma para funcionar sin PostgreSQL/Redis/InfluxDB
"""

from pathlib import Path
import shutil
import sys

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_color(message, color=Colors.GREEN):
    print(f"{color}{message}{Colors.ENDC}")

def setup_lite_mode():
    """Configurar modo LITE automáticamente"""
    
    print_color("\n" + "="*60, Colors.BOLD)
    print_color("   CONFIGURACIÓN MODO LITE (SIN POSTGRESQL)", Colors.BOLD)
    print_color("="*60 + "\n", Colors.BOLD)
    
    try:
        # 1. Crear directorio data
        print_color("📁 Creando directorios necesarios...", Colors.BLUE)
        Path('data').mkdir(exist_ok=True)
        Path('config').mkdir(exist_ok=True)
        print_color("✅ Directorios creados", Colors.GREEN)
        
        # 2. Backup de configuración original
        print_color("\n💾 Creando backups...", Colors.BLUE)
        config_file = Path('config/platform_config.yaml')
        if config_file.exists():
            backup_file = Path('config/platform_config_postgresql.yaml')
            shutil.copy(config_file, backup_file)
            print_color(f"✅ Backup de configuración: {backup_file}", Colors.GREEN)
        
        # 3. Crear configuración LITE
        print_color("\n⚙️  Creando configuración LITE...", Colors.BLUE)
        lite_config = '''# Configuración LITE - Sin PostgreSQL/Redis/InfluxDB
# Ideal para desarrollo y pruebas

platform:
  name: "Trading Platform - Lite Mode"
  version: "1.0.0"
  environment: development
  lite_mode: true

database:
  # SQLite en lugar de PostgreSQL (no requiere instalación)
  postgres_url: "sqlite:///data/trading.db"
  
  # Redis deshabilitado (usar cache en memoria)
  redis_url: "redis://localhost:6379/0"
  redis_enabled: false
  
  # InfluxDB deshabilitado
  influx_url: "http://localhost:8086"
  influx_enabled: false
  influx_token: "dev-token"
  influx_org: "trading"

mt5:
  path: "C:/Program Files/MetaTrader 5/terminal64.exe"
  server: ""
  login: 0
  password: ""
  timeout: 60000
  portable: false

risk:
  max_drawdown: 0.15
  max_position_size: 0.1
  daily_loss_limit: 0.05
  correlation_threshold: 0.7

optimization:
  population_size: 50
  generations: 20
  mutation_rate: 0.1
  crossover_rate: 0.8

monitoring:
  enable_performance_monitoring: true
  enable_trade_monitoring: true
  enable_system_monitoring: true
  log_level: "INFO"
  log_retention_days: 7
  metrics_interval: 60
'''
        
        config_file.write_text(lite_config, encoding='utf-8')
        print_color(f"✅ Configuración LITE creada: {config_file}", Colors.GREEN)
        
        # 4. Patch DataManager
        print_color("\n🔧 Actualizando DataManager...", Colors.BLUE)
        patch_data_manager()
        
        # 5. Crear archivo .env
        print_color("\n📝 Creando archivo .env...", Colors.BLUE)
        create_env_file()
        
        # 6. Verificación
        print_color("\n🔍 Verificando configuración...", Colors.BLUE)
        verify_setup()
        
        # Resumen
        print_color("\n" + "="*60, Colors.BOLD)
        print_color("✅ MODO LITE CONFIGURADO EXITOSAMENTE", Colors.GREEN)
        print_color("="*60, Colors.BOLD)
        
        print_color("\n📊 Configuración aplicada:", Colors.BLUE)
        print("   • Base de datos: SQLite (data/trading.db)")
        print("   • Cache: Memoria (sin Redis)")
        print("   • Time series: Deshabilitado (sin InfluxDB)")
        print("   • Modo: Desarrollo")
        
        print_color("\n📋 Próximos pasos:", Colors.BLUE)
        print("   1. python main.py --health-check")
        print("   2. python main.py --environment development")
        print("   3. python main.py --gui (para interfaz gráfica)")
        
        print_color("\n💡 Notas importantes:", Colors.YELLOW)
        print("   • La base de datos SQLite está en: data/trading.db")
        print("   • Los backups están en: config/platform_config_postgresql.yaml")
        print("   • Para volver a PostgreSQL, restaura el backup")
        
        print_color("\n" + "="*60, Colors.BOLD)
        
        return True
        
    except Exception as e:
        print_color(f"\n❌ Error durante la configuración: {e}", Colors.RED)
        import traceback
        traceback.print_exc()
        return False

def patch_data_manager():
    """Parchear DataManager para soportar modo lite"""
    data_manager_file = Path('database/data_manager.py')
    
    if not data_manager_file.exists():
        print_color("⚠️  database/data_manager.py no encontrado", Colors.YELLOW)
        return
    
    # Backup
    backup_dm = Path('database/data_manager_postgresql.py')
    if not backup_dm.exists():
        shutil.copy(data_manager_file, backup_dm)
        print_color(f"✅ Backup de DataManager: {backup_dm}", Colors.GREEN)
    
    # Leer contenido
    content = data_manager_file.read_text(encoding='utf-8')
    
    # Verificar si ya está parcheado
    if 'LITE MODE PATCH' in content:
        print_color("✅ DataManager ya está configurado para modo lite", Colors.GREEN)
        return
    
    # Agregar import de Path si no existe
    if 'from pathlib import Path' not in content:
        content = 'from pathlib import Path\n' + content
    
    # Insertar código de modo lite
    lite_patch = '''
        # === LITE MODE PATCH ===
        # Soporte para SQLite y cache en memoria
        try:
            db_url = self.config.database.postgres_url
            
            if 'sqlite' in db_url.lower():
                logger.info("🔧 Configurando modo LITE (SQLite + Cache en memoria)")
                
                # Crear directorio data
                Path('data').mkdir(exist_ok=True)
                
                # SQLite
                self.pg_engine = create_engine(
                    db_url,
                    echo=False,
                    connect_args={'check_same_thread': False}
                )
                self.Session = sessionmaker(bind=self.pg_engine)
                
                # Crear tablas
                TradingData.create_table(self.pg_engine)
                logger.info("✅ Base de datos SQLite configurada")
                
                # Cache en memoria (reemplazo de Redis)
                class InMemoryCache:
                    def __init__(self):
                        self.cache = {}
                        self.expiry = {}
                    
                    def get(self, key):
                        from datetime import datetime
                        if key in self.cache:
                            if key in self.expiry:
                                if datetime.now() > self.expiry[key]:
                                    del self.cache[key]
                                    del self.expiry[key]
                                    return None
                            return self.cache[key]
                        return None
                    
                    def setex(self, key, ttl, value):
                        from datetime import datetime, timedelta
                        self.cache[key] = value
                        self.expiry[key] = datetime.now() + timedelta(seconds=ttl)
                    
                    def set(self, key, value):
                        self.cache[key] = value
                    
                    def ping(self):
                        return True
                    
                    def flushdb(self):
                        self.cache.clear()
                        self.expiry.clear()
                
                self.redis_client = InMemoryCache()
                logger.info("✅ Cache en memoria configurado")
                
                # InfluxDB deshabilitado
                self.influx_client = None
                self.write_api = None
                self.query_api = None
                logger.info("ℹ️  InfluxDB deshabilitado (modo lite)")
                
                logger.info("✅ Modo LITE configurado correctamente")
                return
        except Exception as e:
            logger.error(f"Error en configuración modo lite: {e}")
        # === FIN LITE MODE PATCH ===
'''
    
    # Insertar después de "def setup_databases(self):"
    content = content.replace(
        'def setup_databases(self):',
        f'def setup_databases(self):{lite_patch}',
        1  # Solo la primera ocurrencia
    )
    
    # Guardar
    data_manager_file.write_text(content, encoding='utf-8')
    print_color("✅ DataManager actualizado para modo LITE", Colors.GREEN)

def create_env_file():
    """Crear archivo .env para desarrollo"""
    env_file = Path('.env')
    
    if env_file.exists():
        print_color("ℹ️  Archivo .env ya existe (no se sobrescribe)", Colors.YELLOW)
        return
    
    env_content = '''# Configuración de Desarrollo - Modo LITE

# Base de Datos (SQLite)
DATABASE_URL=sqlite:///data/trading.db

# Cache (Deshabilitado - usando memoria)
REDIS_ENABLED=false

# Time Series (Deshabilitado)
INFLUX_ENABLED=false

# Entorno
ENVIRONMENT=development
DEBUG=true

# MetaTrader 5 (Configurar según tu instalación)
MT5_PATH=C:/Program Files/MetaTrader 5/terminal64.exe
MT5_LOGIN=
MT5_PASSWORD=
MT5_SERVER=

# Logging
LOG_LEVEL=INFO
'''
    
    env_file.write_text(env_content, encoding='utf-8')
    print_color(f"✅ Archivo .env creado: {env_file}", Colors.GREEN)

def verify_setup():
    """Verificar que la configuración está correcta"""
    try:
        # Verificar archivo de configuración
        config_file = Path('config/platform_config.yaml')
        if config_file.exists():
            content = config_file.read_text()
            if 'sqlite' in content.lower():
                print_color("✅ Configuración LITE detectada", Colors.GREEN)
            else:
                print_color("⚠️  Configuración no parece ser LITE", Colors.YELLOW)
        
        # Verificar directorio data
        if Path('data').exists():
            print_color("✅ Directorio data existe", Colors.GREEN)
        
        # Intentar importar
        sys.path.insert(0, str(Path.cwd()))
        try:
            from config.settings import ConfigManager
            config = ConfigManager()
            db_url = config.database.postgres_url
            
            if 'sqlite' in db_url.lower():
                print_color(f"✅ ConfigManager usa SQLite: {db_url}", Colors.GREEN)
            else:
                print_color(f"⚠️  ConfigManager NO usa SQLite: {db_url}", Colors.YELLOW)
                
        except Exception as e:
            print_color(f"⚠️  No se pudo verificar ConfigManager: {e}", Colors.YELLOW)
        
    except Exception as e:
        print_color(f"⚠️  Error en verificación: {e}", Colors.YELLOW)

def main():
    """Función principal"""
    print_color("\n🚀 Iniciando configuración de modo LITE...\n", Colors.BLUE)
    
    # Verificar que estamos en el directorio correcto
    if not Path('main.py').exists():
        print_color("❌ Error: main.py no encontrado", Colors.RED)
        print_color("   Ejecuta este script desde el directorio raíz del proyecto", Colors.YELLOW)
        return False
    
    # Ejecutar configuración
    success = setup_lite_mode()
    
    if success:
        print_color("\n🎉 ¡Configuración completada con éxito!", Colors.GREEN)
        print_color("\nEjecuta ahora:", Colors.BLUE)
        print_color("   python main.py --health-check\n", Colors.BOLD)
        return True
    else:
        print_color("\n❌ La configuración falló", Colors.RED)
        print_color("Revisa los errores arriba\n", Colors.YELLOW)
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_color("\n\n❌ Cancelado por el usuario", Colors.RED)
        sys.exit(1)
    except Exception as e:
        print_color(f"\n\n❌ Error crítico: {e}", Colors.RED)
        import traceback
        traceback.print_exc()
        sys.exit(1)