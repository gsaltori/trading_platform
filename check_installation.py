# check_installation.py
import importlib

def check_import(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

print("🔍 Verificando instalación...")

# Paquetes críticos
critical_packages = [
    'pandas', 'numpy', 'sklearn', 'tensorflow',
    'PyQt6', 'matplotlib', 'plotly', 'MetaTrader5',
    'yfinance', 'psutil', 'pyyaml'
]

all_ok = True
for package in critical_packages:
    if check_import(package):
        print(f"✅ {package}")
    else:
        print(f"❌ {package}")
        all_ok = False

if all_ok:
    print("\n🎉 ¡Todas las dependencias están instaladas correctamente!")
    print("   Puedes ejecutar la plataforma.")
else:
    print("\n⚠️  Algunas dependencias faltan. La plataforma puede tener funcionalidades limitadas.")

# Verificar estructura de archivos
import os
required_dirs = ['config', 'data', 'logs', 'strategies']
for dir_name in required_dirs:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"📁 Directorio creado: {dir_name}")

print("\n✅ Verificación completada")