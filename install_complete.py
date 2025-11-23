# install_complete.py - Script de Instalación Completo y Robusto
"""
Script de instalación mejorado para la plataforma de trading
Maneja todas las dependencias, alternativas y configuración inicial
"""

import sys
import subprocess
import os
import platform
from pathlib import Path
import importlib.util
import json

class Colors:
    """Colores para output en terminal"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message, color=Colors.OKGREEN):
    """Imprimir mensaje con color"""
    print(f"{color}{message}{Colors.ENDC}")

def check_python_version():
    """Verificar versión de Python"""
    print_colored("\n📋 Verificando versión de Python...", Colors.OKBLUE)
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print_colored("❌ Python 3.10+ es requerido", Colors.FAIL)
        print(f"   Versión actual: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print_colored(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK", Colors.OKGREEN)
    return True

def check_system_requirements():
    """Verificar requisitos del sistema"""
    print_colored("\n📋 Verificando requisitos del sistema...", Colors.OKBLUE)
    
    import psutil
    
    # Verificar RAM
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    if total_ram_gb < 8:
        print_colored(f"⚠️  RAM: {total_ram_gb:.1f}GB (recomendado: 16GB)", Colors.WARNING)
    else:
        print_colored(f"✅ RAM: {total_ram_gb:.1f}GB - OK", Colors.OKGREEN)
    
    # Verificar espacio en disco
    disk_free_gb = psutil.disk_usage('.').free / (1024**3)
    if disk_free_gb < 10:
        print_colored(f"⚠️  Espacio libre: {disk_free_gb:.1f}GB (mínimo: 10GB)", Colors.WARNING)
    else:
        print_colored(f"✅ Espacio libre: {disk_free_gb:.1f}GB - OK", Colors.OKGREEN)
    
    # Verificar sistema operativo
    os_name = platform.system()
    print_colored(f"ℹ️  Sistema operativo: {os_name}", Colors.OKCYAN)
    
    return True

def upgrade_pip():
    """Actualizar pip"""
    print_colored("\n🔧 Actualizando pip...", Colors.OKBLUE)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print_colored("✅ pip actualizado", Colors.OKGREEN)
        return True
    except:
        print_colored("❌ Error actualizando pip", Colors.FAIL)
        return False

def install_package(package_spec, fallback_spec=None, optional=False):
    """Instalar paquete con fallback y manejo de errores"""
    try:
        print(f"  📦 Instalando {package_spec}...", end=" ")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_spec],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print_colored("✅", Colors.OKGREEN)
        return True
    except subprocess.CalledProcessError:
        if fallback_spec:
            print(f"🔄 Intentando fallback: {fallback_spec}...", end=" ")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", fallback_spec],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print_colored("✅", Colors.OKGREEN)
                return True
            except:
                if not optional:
                    print_colored("❌", Colors.FAIL)
                    return False
                else:
                    print_colored("⚠️  (opcional)", Colors.WARNING)
                    return True
        else:
            if not optional:
                print_colored("❌", Colors.FAIL)
                return False
            else:
                print_colored("⚠️  (opcional)", Colors.WARNING)
                return True

def install_core_packages():
    """Instalar paquetes core"""
    print_colored("\n📦 Instalando paquetes core...", Colors.OKBLUE)
    
    packages = [
        # Data Science
        ("numpy==1.26.0", "numpy"),
        ("pandas==2.1.4", "pandas"),
        ("scikit-learn==1.3.2", "scikit-learn"),
        
        # Numerical
        ("numba==0.58.1", "numba"),
        ("scipy==1.11.4", "scipy"),
        
        # Utilities
        ("psutil==5.9.6", None),
        ("pyyaml==6.0.1", None),
        ("joblib==1.3.2", None),
        ("tqdm==4.66.1", None),
    ]
    
    success_count = 0
    for package, fallback in packages:
        if install_package(package, fallback):
            success_count += 1
    
    print_colored(f"\n✅ {success_count}/{len(packages)} paquetes core instalados", Colors.OKGREEN)
    return success_count == len(packages)

def install_trading_packages():
    """Instalar paquetes de trading"""
    print_colored("\n📦 Instalando paquetes de trading...", Colors.OKBLUE)
    
    packages = [
        ("MetaTrader5==5.0.5370", "MetaTrader5"),
        ("yfinance==0.2.18", "yfinance"),
        ("ccxt==4.1.77", "ccxt"),
    ]
    
    success_count = 0
    for package, fallback in packages:
        if install_package(package, fallback):
            success_count += 1
    
    print_colored(f"\n✅ {success_count}/{len(packages)} paquetes de trading instalados", Colors.OKGREEN)
    return True  # No crítico si fallan

def install_ml_packages():
    """Instalar paquetes de Machine Learning"""
    print_colored("\n📦 Instalando paquetes de ML...", Colors.OKBLUE)
    
    packages = [
        ("tensorflow-cpu==2.16.2", "tensorflow-cpu==2.15.0", True),
        ("xgboost==2.0.2", "xgboost", True),
        ("lightgbm==4.1.0", "lightgbm", True),
    ]
    
    success_count = 0
    for package, fallback, optional in packages:
        if install_package(package, fallback, optional):
            success_count += 1
    
    print_colored(f"\n✅ {success_count}/{len(packages)} paquetes ML instalados", Colors.OKGREEN)
    return True

def install_gui_packages():
    """Instalar paquetes de GUI"""
    print_colored("\n📦 Instalando paquetes de GUI...", Colors.OKBLUE)
    
    packages = [
        ("PyQt6==6.6.1", "PyQt6"),
        ("PyQt6-WebEngine==6.6.0", None, True),
        ("qt-material==2.14", None, True),
    ]
    
    success_count = 0
    for item in packages:
        package = item[0]
        fallback = item[1] if len(item) > 1 else None
        optional = item[2] if len(item) > 2 else False
        
        if install_package(package, fallback, optional):
            success_count += 1
    
    print_colored(f"\n✅ {success_count}/{len(packages)} paquetes GUI instalados", Colors.OKGREEN)
    return True

def install_visualization_packages():
    """Instalar paquetes de visualización"""
    print_colored("\n📦 Instalando paquetes de visualización...", Colors.OKBLUE)
    
    packages = [
        ("matplotlib==3.8.2", "matplotlib"),
        ("plotly==5.17.0", "plotly"),
        ("seaborn==0.13.0", "seaborn"),
    ]
    
    success_count = 0
    for package, fallback in packages:
        if install_package(package, fallback):
            success_count += 1
    
    print_colored(f"\n✅ {success_count}/{len(packages)} paquetes de visualización instalados", Colors.OKGREEN)
    return True

def install_database_packages():
    """Instalar paquetes de base de datos"""
    print_colored("\n📦 Instalando paquetes de base de datos...", Colors.OKBLUE)
    
    packages = [
        ("sqlalchemy==2.0.23", "sqlalchemy"),
        ("redis==5.0.1", "redis"),
        ("psycopg2-binary==2.9.9", "psycopg2-binary"),
        ("influxdb-client==1.39.0", "influxdb-client"),
    ]
    
    success_count = 0
    for package, fallback in packages:
        if install_package(package, fallback):
            success_count += 1
    
    print_colored(f"\n✅ {success_count}/{len(packages)} paquetes de base de datos instalados", Colors.OKGREEN)
    return True

def install_optimization_packages():
    """Instalar paquetes de optimización"""
    print_colored("\n📦 Instalando paquetes de optimización...", Colors.OKBLUE)
    
    packages = [
        ("bayesian-optimization==1.4.3", "bayesian-optimization", True),
        ("deap==1.4.1", "deap", True),
    ]
    
    success_count = 0
    for package, fallback, optional in packages:
        if install_package(package, fallback, optional):
            success_count += 1
    
    print_colored(f"\n✅ {success_count}/{len(packages)} paquetes de optimización instalados", Colors.OKGREEN)
    return True

def install_ta_lib():
    """Intentar instalar TA-Lib con alternativas"""
    print_colored("\n📦 Instalando indicadores técnicos...", Colors.OKBLUE)
    
    alternatives = [
        ("TA-Lib", "Versión nativa C (recomendada)"),
        ("ta", "Alternativa Python pura"),
        ("pandas-ta", "Alternativa basada en pandas"),
        ("tulipy", "Alternativa ligera"),
    ]
    
    for package, description in alternatives:
        try:
            print(f"  📦 Intentando {package} ({description})...", end=" ")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print_colored("✅", Colors.OKGREEN)
            print_colored(f"  ✅ Instalado: {package}", Colors.OKGREEN)
            return True
        except:
            print_colored("❌", Colors.FAIL)
            continue
    
    print_colored("  ⚠️  No se pudo instalar ninguna alternativa de TA-Lib", Colors.WARNING)
    print_colored("     La plataforma funcionará con indicadores limitados", Colors.WARNING)
    return True  # No crítico

def create_directory_structure():
    """Crear estructura de directorios"""
    print_colored("\n📁 Creando estructura de directorios...", Colors.OKBLUE)
    
    directories = [
        'config',
        'logs',
        'data/cache',
        'data/backtests',
        'data/backups',
        'strategies/generated',
        'strategies/optimized',
        'reports/html',
        'reports/pdf',
        'ml/models',
        'ml/datasets',
        'tests',
        'docs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")
    
    print_colored("✅ Estructura de directorios creada", Colors.OKGREEN)
    return True

def create_config_files():
    """Crear archivos de configuración iniciales"""
    print_colored("\n⚙️  Creando archivos de configuración...", Colors.OKBLUE)
    
    # Archivo .env.example
    env_example = """# Configuración de la Plataforma de Trading

# Base de Datos
POSTGRES_URL=postgresql://trading_user:trading_password@localhost:5432/trading
REDIS_URL=redis://localhost:6379/0
INFLUX_URL=http://localhost:8086
INFLUX_TOKEN=your-influx-token
INFLUX_ORG=trading

# MetaTrader 5
MT5_PATH=C:/Program Files/MetaTrader 5/terminal64.exe
MT5_SERVER=YourBroker-Demo
MT5_LOGIN=12345678
MT5_PASSWORD=your-password

# Alertas
EMAIL_ENABLED=false
EMAIL_FROM=trading@example.com
EMAIL_TO=alerts@example.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

TELEGRAM_ENABLED=false
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id

# AWS S3 (opcional)
S3_ENABLED=false
S3_BUCKET=trading-backups
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_ENABLED=true
GRAFANA_PORT=3000

# Trading
ENVIRONMENT=development
MAX_POSITIONS=5
RISK_PER_TRADE=0.02
DAILY_LOSS_LIMIT=0.05
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_example)
    print("  ✅ .env.example")
    
    # Archivo de configuración YAML
    config_yaml = """# Platform Configuration
platform:
  name: "Trading Platform"
  version: "1.0.0"
  environment: development

database:
  postgres_url: "postgresql://trading_user:trading_password@localhost:5432/trading"
  redis_url: "redis://localhost:6379/0"
  influx_url: "http://localhost:8086"
  influx_token: "your-token"
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
  population_size: 100
  generations: 50
  mutation_rate: 0.1
  crossover_rate: 0.8

monitoring:
  enable_performance_monitoring: true
  enable_trade_monitoring: true
  enable_system_monitoring: true
  log_level: "INFO"
  log_retention_days: 30
  metrics_interval: 60
"""
    
    with open('config/platform_config.yaml', 'w') as f:
        f.write(config_yaml)
    print("  ✅ config/platform_config.yaml")
    
    print_colored("✅ Archivos de configuración creados", Colors.OKGREEN)
    return True

def verify_installation():
    """Verificar la instalación"""
    print_colored("\n🧪 Verificando instalación...", Colors.OKBLUE)
    
    critical_packages = [
        'pandas', 'numpy', 'sklearn', 'numba',
        'matplotlib', 'plotly', 'sqlalchemy', 'redis'
    ]
    
    optional_packages = [
        'MetaTrader5', 'PyQt6', 'tensorflow', 'xgboost'
    ]
    
    all_ok = True
    
    # Verificar paquetes críticos
    print("\nPaquetes críticos:")
    for package in critical_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print_colored(f"  ❌ {package} - NO IMPORTA", Colors.FAIL)
            all_ok = False
    
    # Verificar paquetes opcionales
    print("\nPaquetes opcionales:")
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print_colored(f"  ⚠️  {package} - no disponible (opcional)", Colors.WARNING)
    
    return all_ok

def generate_installation_report():
    """Generar reporte de instalación"""
    print_colored("\n📊 Generando reporte de instalación...", Colors.OKBLUE)
    
    report = {
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'platform': platform.system(),
        'architecture': platform.machine(),
        'installed_packages': []
    }
    
    # Lista de paquetes instalados
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True
        )
        report['installed_packages'] = json.loads(result.stdout)
    except:
        pass
    
    # Guardar reporte
    report_path = 'logs/installation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print_colored(f"✅ Reporte guardado en: {report_path}", Colors.OKGREEN)
    return True

def main():
    """Función principal de instalación"""
    print_colored("\n" + "="*60, Colors.HEADER)
    print_colored("   INSTALACIÓN DE PLATAFORMA DE TRADING ALGORÍTMICO", Colors.HEADER)
    print_colored("="*60 + "\n", Colors.HEADER)
    
    # 1. Verificar Python
    if not check_python_version():
        return False
    
    # 2. Verificar requisitos del sistema
    check_system_requirements()
    
    # 3. Actualizar pip
    if not upgrade_pip():
        print_colored("⚠️  Continuando sin actualizar pip...", Colors.WARNING)
    
    # 4. Instalar paquetes
    steps = [
        ("Core", install_core_packages),
        ("Trading", install_trading_packages),
        ("ML", install_ml_packages),
        ("GUI", install_gui_packages),
        ("Visualización", install_visualization_packages),
        ("Base de Datos", install_database_packages),
        ("Optimización", install_optimization_packages),
        ("TA-Lib", install_ta_lib),
    ]
    
    for step_name, step_func in steps:
        try:
            if not step_func():
                print_colored(f"⚠️  Advertencia en: {step_name}", Colors.WARNING)
        except Exception as e:
            print_colored(f"❌ Error en {step_name}: {e}", Colors.FAIL)
    
    # 5. Crear estructura
    create_directory_structure()
    
    # 6. Crear configuraciones
    create_config_files()
    
    # 7. Verificar instalación
    if verify_installation():
        print_colored("\n✅ INSTALACIÓN COMPLETADA EXITOSAMENTE", Colors.OKGREEN)
    else:
        print_colored("\n⚠️  INSTALACIÓN COMPLETADA CON ADVERTENCIAS", Colors.WARNING)
    
    # 8. Generar reporte
    generate_installation_report()
    
    # 9. Próximos pasos
    print_colored("\n📋 PRÓXIMOS PASOS:", Colors.OKBLUE)
    print("   1. Copiar .env.example a .env y configurar tus credenciales")
    print("   2. Ajustar config/platform_config.yaml según tus necesidades")
    print("   3. Ejecutar: python main.py --environment development")
    print("   4. Para la interfaz gráfica: python main.py --gui")
    print("   5. Para tests: python -m pytest tests/")
    
    print_colored("\n" + "="*60, Colors.HEADER)
    print_colored("   ¡Instalación completada! Happy Trading! 🚀", Colors.HEADER)
    print_colored("="*60 + "\n", Colors.HEADER)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_colored("\n\n❌ Instalación cancelada por el usuario", Colors.FAIL)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n\n❌ Error crítico: {e}", Colors.FAIL)
        import traceback
        traceback.print_exc()
        sys.exit(1)