# install_robust.py
import sys
import subprocess
import importlib.util

def install_package(package_spec, fallback_spec=None):
    """Instalar paquete con fallback"""
    try:
        print(f"📦 Instalando {package_spec}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        print(f"✅ {package_spec} instalado")
        return True
    except subprocess.CalledProcessError:
        if fallback_spec:
            print(f"🔄 Fallando a {fallback_spec}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", fallback_spec])
                print(f"✅ {fallback_spec} instalado")
                return True
            except subprocess.CalledProcessError:
                print(f"❌ No se pudo instalar {package_spec} ni {fallback_spec}")
                return False
        else:
            print(f"❌ No se pudo instalar {package_spec}")
            return False

def main():
    print("🚀 Instalación robusta de la plataforma de trading...")
    
    # Actualizar pip primero
    install_package("--upgrade pip")
    
    # Paquetes críticos con fallbacks
    packages = [
        # Data Science
        ("numpy==1.26.0", "numpy"),
        ("pandas==2.1.4", "pandas"),
        ("scikit-learn==1.3.2", "scikit-learn"),
        ("tensorflow-cpu==2.16.2", "tensorflow-cpu"),
        
        # GUI
        ("PyQt6==6.6.1", None),
        ("PyQt6-WebEngine==6.6.0", None),
        ("qt-material==2.14", None),
        
        # Trading
        ("MetaTrader5==5.0.5370", "MetaTrader5"),
        ("yfinance==0.2.18", "yfinance"),
        ("ccxt==4.1.77", "ccxt"),
        
        # Visualización
        ("plotly==5.17.0", "plotly"),
        ("matplotlib==3.8.2", "matplotlib"),
        ("seaborn==0.13.0", "seaborn"),
        
        # Base de datos
        ("sqlalchemy==2.0.23", "sqlalchemy"),
        ("redis==5.0.1", "redis"),
        ("psycopg2-binary==2.9.9", "psycopg2-binary"),
        
        # Utilidades
        ("psutil==5.9.6", "psutil"),
        ("pyyaml==6.0.1", "pyyaml"),
        ("joblib==1.3.2", "joblib"),
        ("tqdm==4.66.1", "tqdm"),
    ]
    
    print("\n🔧 Instalando paquetes críticos...")
    success_count = 0
    for package, fallback in packages:
        if install_package(package, fallback):
            success_count += 1
    
    # Intentar TA-Lib alternativo
    print("\n🔄 Intentando alternativas para TA-Lib...")
    ta_alternatives = [
        "TA-Lib",  # Nombre correcto del paquete
        "pandas-ta",  # Alternativa en Python puro
        "tulipy",  # Otra alternativa
    ]
    
    for ta_package in ta_alternatives:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", ta_package])
            print(f"✅ {ta_package} instalado como alternativa a TA-Lib")
            break
        except subprocess.CalledProcessError:
            continue
    else:
        print("⚠️  No se pudo instalar TA-Lib ni alternativas")
        print("   La plataforma funcionará pero algunos indicadores técnicos no estarán disponibles")
    
    # Paquetes opcionales
    optional_packages = [
        "xgboost==2.0.2",
        "lightgbm==4.1.0", 
        "bayesian-optimization==1.4.3",
        "deap==1.4.1",
        "influxdb-client==1.39.0",
        "docker==6.1.3",
        "pytest==7.4.3",
        "black==23.9.1",
        "flake8==6.1.0",
        "mypy==1.6.1"
    ]
    
    print("\n📦 Instalando paquetes opcionales...")
    for package in optional_packages:
        install_package(package)
    
    print(f"\n🎉 Instalación completada: {success_count}/{len(packages)} paquetes críticos instalados")
    
    # Verificar instalación
    print("\n🧪 Verificando instalación...")
    test_imports = [
        'pandas', 'numpy', 'sklearn', 'tensorflow',
        'PyQt6', 'matplotlib', 'plotly', 'MetaTrader5'
    ]
    
    for package in test_imports:
        try:
            importlib.import_module(package)
            print(f"✅ {package} importa correctamente")
        except ImportError as e:
            print(f"❌ Error importando {package}: {e}")

if __name__ == "__main__":
    main()