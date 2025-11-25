#!/usr/bin/env python3
"""
Diagnóstico completo de PyQt6 y dependencias
"""

import sys
import subprocess
import platform

def check_pyqt6():
    """Diagnosticar problemas de PyQt6"""
    
    print("🔍 DIAGNÓSTICO DE PyQt6")
    print("="*70)
    
    # Información del sistema
    print(f"\n📊 Sistema:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version}")
    print(f"   Arquitectura: {platform.machine()}")
    
    # Verificar PyQt6 instalado
    print(f"\n📦 Paquetes PyQt6:")
    packages = ['PyQt6', 'PyQt6-Qt6', 'PyQt6-sip', 'PyQt6-WebEngine']
    for package in packages:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                version_line = [l for l in result.stdout.split('\n') if l.startswith('Version:')]
                version = version_line[0].split(':')[1].strip() if version_line else 'Unknown'
                location_line = [l for l in result.stdout.split('\n') if l.startswith('Location:')]
                location = location_line[0].split(':')[1].strip() if location_line else 'Unknown'
                print(f"   ✅ {package}: {version}")
                print(f"      Ubicación: {location}")
            else:
                print(f"   ❌ {package}: No instalado")
        except Exception as e:
            print(f"   ❌ {package}: Error - {e}")
    
    # Intentar importar
    print(f"\n🧪 Pruebas de importación:")
    
    import_tests = [
        ("PyQt6", "import PyQt6"),
        ("PyQt6.QtCore", "from PyQt6 import QtCore"),
        ("PyQt6.QtWidgets", "from PyQt6 import QtWidgets"),
        ("PyQt6.QtGui", "from PyQt6 import QtGui"),
    ]
    
    for name, import_cmd in import_tests:
        try:
            result = subprocess.run(
                [sys.executable, "-c", import_cmd],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                print(f"   ✅ {name}: OK")
            else:
                print(f"   ❌ {name}: FALLO")
                if result.stderr:
                    print(f"      Error: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"   ⏱️  {name}: TIMEOUT")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    # Verificar DLLs de sistema
    print(f"\n🔧 Verificando dependencias del sistema:")
    
    # Visual C++ Redistributable
    try:
        import winreg
        vcredist_keys = [
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
            r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        ]
        
        vcredist_found = False
        for key_path in vcredist_keys:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                vcredist_found = True
                print(f"   ✅ Visual C++ Redistributable detectado")
                winreg.CloseKey(key)
                break
            except FileNotFoundError:
                continue
        
        if not vcredist_found:
            print(f"   ❌ Visual C++ Redistributable NO encontrado")
            print(f"      Descargar: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    except ImportError:
        print(f"   ⚠️  No se puede verificar en este sistema (no Windows)")
    except Exception as e:
        print(f"   ⚠️  Error verificando VC++ Redistributable: {e}")
    
    # PATH
    print(f"\n📁 Variables de entorno:")
    path = sys.path
    print(f"   Python paths: {len(path)} directorios")
    for p in path[:3]:
        print(f"      - {p}")
    if len(path) > 3:
        print(f"      ... y {len(path)-3} más")
    
    # Resumen
    print(f"\n" + "="*70)
    print(f"📋 DIAGNÓSTICO COMPLETO")
    
    # Intentar importar PyQt6 completamente
    try:
        exec("from PyQt6.QtWidgets import QApplication")
        print(f"\n✅ PyQt6 está funcionando correctamente!")
        print(f"\nPuedes ejecutar:")
        print(f"   python main.py --gui")
    except Exception as e:
        print(f"\n❌ PyQt6 tiene problemas:")
        print(f"   {type(e).__name__}: {str(e)[:200]}")
        
        print(f"\n💡 SOLUCIONES RECOMENDADAS:")
        
        if "DLL load failed" in str(e):
            print(f"\n1. INSTALAR VISUAL C++ REDISTRIBUTABLE (MÁS PROBABLE):")
            print(f"   - Descargar: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print(f"   - Instalar y reiniciar el sistema")
            print(f"   - Volver a intentar")
            
            print(f"\n2. REINSTALAR PyQt6:")
            print(f"   python /mnt/user-data/outputs/fix_pyqt6.py")
            
            print(f"\n3. USAR MODO HEADLESS (temporal):")
            print(f"   python main.py --headless")
        else:
            print(f"\n1. Reinstalar PyQt6:")
            print(f"   python /mnt/user-data/outputs/fix_pyqt6.py")
            
            print(f"\n2. Verificar instalación de Python")
            print(f"\n3. Usar modo headless:")
            print(f"   python main.py --headless")

if __name__ == "__main__":
    check_pyqt6()