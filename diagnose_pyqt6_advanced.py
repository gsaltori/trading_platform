#!/usr/bin/env python3
"""
Diagnóstico avanzado de PyQt6 - Resolver problemas de DLL después de instalar VC++ Redistributable
"""

import sys
import subprocess
import os
import platform

def check_vcredist_installed():
    """Verificar si Visual C++ Redistributable está realmente instalado"""
    print("🔍 Verificando Visual C++ Redistributable...")
    print("-"*70)
    
    try:
        import winreg
        
        # Claves de registro donde se instala VC++ Redistributable
        keys_to_check = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x86"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"),
            (winreg.HKEY_CLASSES_ROOT, r"Installer\Dependencies\VC,redist.x64,amd64,14.34,bundle"),
            (winreg.HKEY_CLASSES_ROOT, r"Installer\Dependencies\VC,redist.x64,amd64,14.38,bundle"),
        ]
        
        found_versions = []
        
        for root, key_path in keys_to_check:
            try:
                key = winreg.OpenKey(root, key_path)
                try:
                    version, _ = winreg.QueryValueEx(key, "Version")
                    found_versions.append((key_path, version))
                    print(f"   ✅ Encontrado: {key_path}")
                    print(f"      Versión: {version}")
                except:
                    pass
                winreg.CloseKey(key)
            except FileNotFoundError:
                continue
        
        if found_versions:
            print(f"\n✅ Visual C++ Redistributable INSTALADO")
            print(f"   Versiones encontradas: {len(found_versions)}")
            return True
        else:
            print(f"\n❌ Visual C++ Redistributable NO encontrado en registro")
            return False
            
    except Exception as e:
        print(f"⚠️  Error verificando registro: {e}")
        return None

def check_python_architecture():
    """Verificar arquitectura de Python"""
    print("\n💻 Verificando Arquitectura...")
    print("-"*70)
    
    is_64bit = sys.maxsize > 2**32
    print(f"   Python: {'64-bit' if is_64bit else '32-bit'}")
    print(f"   OS: {platform.machine()}")
    print(f"   Plataforma: {platform.platform()}")
    
    if not is_64bit:
        print("\n⚠️  ADVERTENCIA: Python es 32-bit")
        print("   PyQt6 funciona mejor con Python 64-bit")
        print("   Considera reinstalar Python 64-bit")
        return False
    
    return True

def check_system_path():
    """Verificar PATH del sistema"""
    print("\n📁 Verificando System32 en PATH...")
    print("-"*70)
    
    path = os.environ.get('PATH', '').split(';')
    system32_found = False
    
    system32_paths = [
        r'C:\Windows\System32',
        r'C:\Windows\SysWOW64',
    ]
    
    for sys_path in system32_paths:
        if any(sys_path.lower() in p.lower() for p in path):
            print(f"   ✅ {sys_path} en PATH")
            system32_found = True
        else:
            print(f"   ❌ {sys_path} NO en PATH")
    
    return system32_found

def try_import_pyqt6():
    """Intentar importar PyQt6 con diagnóstico detallado"""
    print("\n🧪 Intentando importar PyQt6...")
    print("-"*70)
    
    tests = [
        ("PyQt6", "import PyQt6"),
        ("PyQt6.QtCore", "from PyQt6 import QtCore"),
        ("PyQt6.QtWidgets", "from PyQt6.QtWidgets import QApplication"),
    ]
    
    for name, import_cmd in tests:
        try:
            print(f"   Probando: {name}...")
            exec(import_cmd)
            print(f"   ✅ {name} importado correctamente")
        except ImportError as e:
            print(f"   ❌ {name} FALLÓ")
            print(f"      Error: {str(e)[:150]}")
            return False
        except Exception as e:
            print(f"   ❌ {name} ERROR INESPERADO")
            print(f"      Error: {str(e)[:150]}")
            return False
    
    return True

def get_pyqt6_info():
    """Obtener información de PyQt6 instalado"""
    print("\n📦 Información de PyQt6...")
    print("-"*70)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "PyQt6"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:') or line.startswith('Location:'):
                    print(f"   {line}")
            
            # Verificar archivos DLL de PyQt6
            try:
                import PyQt6
                from pathlib import Path
                
                pyqt6_path = Path(PyQt6.__file__).parent
                qt6_bin = pyqt6_path / 'Qt6' / 'bin'
                
                if qt6_bin.exists():
                    print(f"\n   PyQt6 DLLs en: {qt6_bin}")
                    
                    critical_dlls = ['Qt6Core.dll', 'Qt6Gui.dll', 'Qt6Widgets.dll']
                    for dll in critical_dlls:
                        dll_path = qt6_bin / dll
                        if dll_path.exists():
                            size_mb = dll_path.stat().st_size / 1024 / 1024
                            print(f"   ✅ {dll} ({size_mb:.1f} MB)")
                        else:
                            print(f"   ❌ {dll} FALTA")
                else:
                    print(f"   ⚠️  Directorio Qt6/bin no encontrado")
            except:
                pass
        
    except Exception as e:
        print(f"   ⚠️  Error obteniendo info: {e}")

def main():
    print("="*70)
    print("🔧 DIAGNÓSTICO AVANZADO DE PyQt6")
    print("="*70)
    
    # Verificaciones
    vcredist_ok = check_vcredist_installed()
    arch_ok = check_python_architecture()
    path_ok = check_system_path()
    get_pyqt6_info()
    pyqt6_ok = try_import_pyqt6()
    
    # Resumen
    print("\n" + "="*70)
    print("📊 RESUMEN DEL DIAGNÓSTICO")
    print("="*70)
    
    checks = [
        ("Visual C++ Redistributable", vcredist_ok),
        ("Python 64-bit", arch_ok),
        ("System32 en PATH", path_ok),
        ("PyQt6 Importable", pyqt6_ok),
    ]
    
    for check_name, status in checks:
        if status is True:
            print(f"   ✅ {check_name}")
        elif status is False:
            print(f"   ❌ {check_name}")
        else:
            print(f"   ⚠️  {check_name} (no determinado)")
    
    # Soluciones
    print("\n" + "="*70)
    print("💡 SOLUCIONES RECOMENDADAS")
    print("="*70)
    
    if not pyqt6_ok:
        print("\n🔧 SOLUCIÓN 1: Reinstalar PyQt6 COMPLETAMENTE")
        print("-"*70)
        print("Ejecuta estos comandos en orden:")
        print()
        print("1. Desinstalar completamente:")
        print("   pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip PyQt6-WebEngine")
        print()
        print("2. Limpiar cache:")
        print("   pip cache purge")
        print()
        print("3. Reinstalar desde cero:")
        print("   pip install --no-cache-dir PyQt6==6.6.1")
        print()
        print("4. Verificar:")
        print("   python -c \"from PyQt6.QtWidgets import QApplication; print('OK')\"")
        
        print("\n🔧 SOLUCIÓN 2: Probar con PyQt5 (alternativa)")
        print("-"*70)
        print("PyQt5 es más estable en algunos sistemas Windows:")
        print()
        print("1. Instalar PyQt5:")
        print("   pip install PyQt5")
        print()
        print("2. Modificar imports en ui/main_window.py")
        print("   Cambiar: from PyQt6 import ...")
        print("   Por:     from PyQt5 import ...")
        
        print("\n🔧 SOLUCIÓN 3: Verificar versión de Windows")
        print("-"*70)
        print(f"Tu Windows: {platform.platform()}")
        print()
        print("PyQt6 requiere:")
        print("   - Windows 10 versión 1809 o superior")
        print("   - Windows 11")
        print()
        print("Si tienes Windows más antiguo, usa PyQt5")
        
        print("\n🔧 SOLUCIÓN 4: Usar modo HEADLESS (recomendado temporalmente)")
        print("-"*70)
        print("Mientras resuelves PyQt6, usa la plataforma sin GUI:")
        print()
        print("   python main.py --headless")
        print()
        print("Esto te da acceso completo a:")
        print("   ✅ Backtesting")
        print("   ✅ Machine Learning")
        print("   ✅ Optimización")
        print("   ✅ Live Trading")
        print("   ✅ Todas las funcionalidades excepto GUI")
        
        if not arch_ok:
            print("\n🔧 SOLUCIÓN 5: Instalar Python 64-bit")
            print("-"*70)
            print("Tu Python actual es 32-bit")
            print()
            print("1. Descargar Python 64-bit:")
            print("   https://www.python.org/downloads/")
            print()
            print("2. Durante instalación, seleccionar:")
            print("   ☑ Add Python to PATH")
            print("   ☑ Install for all users")
            print()
            print("3. Reinstalar todas las dependencias")
        
        if vcredist_ok == False:
            print("\n🔧 SOLUCIÓN 6: Reinstalar Visual C++ Redistributable")
            print("-"*70)
            print("El registro no muestra VC++ Redistributable instalado")
            print()
            print("1. Descargar AMBAS versiones:")
            print("   x64: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("   x86: https://aka.ms/vs/17/release/vc_redist.x86.exe")
            print()
            print("2. Instalar x64 primero, luego x86")
            print("3. Reiniciar el sistema")
            print("4. Ejecutar este diagnóstico nuevamente")
    
    else:
        print("\n✅ PyQt6 está funcionando correctamente!")
        print("\nPuedes ejecutar:")
        print("   python main.py --gui")
    
    print("\n" + "="*70)
    print("📞 SIGUIENTE PASO")
    print("="*70)
    
    if not pyqt6_ok:
        print("\nMi recomendación: Usar modo HEADLESS mientras resuelves PyQt6")
        print()
        print("Ejecuta ahora:")
        print("   python main.py --headless")
        print()
        print("Tienes acceso completo a todas las funcionalidades")
        print("La GUI es solo una interfaz visual adicional")
    
    return pyqt6_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)