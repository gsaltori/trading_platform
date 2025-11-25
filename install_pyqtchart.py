#!/usr/bin/env python3
"""
Solución completa para PyQt5: Instalar PyQtChart y arreglar imports
"""

import subprocess
import sys

def install_pyqtchart():
    """Instalar PyQtChart para PyQt5"""
    
    print("🔧 SOLUCIÓN COMPLETA PARA PyQt5")
    print("="*70)
    
    print("\n📦 Paso 1: Instalando PyQtChart...")
    print("-"*70)
    
    try:
        # Instalar PyQtChart
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "PyQtChart"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print("✅ PyQtChart instalado correctamente")
        else:
            print("⚠️  Advertencia durante instalación:")
            print(result.stderr[:300])
    except Exception as e:
        print(f"❌ Error instalando PyQtChart: {e}")
        return False
    
    print("\n🔧 Paso 2: Arreglando imports en ui/main_window.py...")
    print("-"*70)
    
    try:
        file_path = 'ui/main_window.py'
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Cambiar QtCharts por QtChart (el nombre correcto en PyQt5)
        old_import = "from PyQt5.QtCharts import"
        new_import = "from PyQtChart.QtChart import"
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            print("✅ Cambiado PyQt5.QtCharts -> PyQtChart.QtChart")
        
        # Escribir cambios
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Archivo actualizado")
        
    except FileNotFoundError:
        print(f"❌ No se encontró {file_path}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n🧪 Paso 3: Verificando instalación...")
    print("-"*70)
    
    # Verificar imports
    tests = [
        ("PyQt5.QtWidgets", "from PyQt5.QtWidgets import QApplication"),
        ("PyQt5.QtCore", "from PyQt5.QtCore import Qt"),
        ("PyQtChart", "from PyQtChart.QtChart import QChart"),
    ]
    
    all_ok = True
    for name, import_cmd in tests:
        try:
            exec(import_cmd)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {str(e)[:100]}")
            all_ok = False
    
    print("\n" + "="*70)
    if all_ok:
        print("✅ TODO LISTO!")
        print("="*70)
        print("\nAhora ejecuta:")
        print("   python main.py --gui")
        return True
    else:
        print("⚠️  ALGUNOS PROBLEMAS PERSISTEN")
        print("="*70)
        print("\nAlternativa recomendada:")
        print("   python /mnt/user-data/outputs/simple_gui.py")
        return False

if __name__ == "__main__":
    success = install_pyqtchart()
    
    if not success:
        print("\n" + "="*70)
        print("💡 OTRAS OPCIONES")
        print("="*70)
        
        print("\n1. GUI Simple con Tkinter (SIN PyQt):")
        print("   python /mnt/user-data/outputs/simple_gui.py")
        
        print("\n2. Modo Headless (sin GUI):")
        print("   python main.py --headless")
        
        print("\n3. Volver a PyQt6:")
        print("   pip uninstall -y PyQt5 PyQtChart")
        print("   pip install PyQt6==6.6.1")
        print("   # Luego arreglar imports de vuelta a PyQt6")