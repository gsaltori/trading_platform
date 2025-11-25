#!/usr/bin/env python3
"""
Detectar qué DLLs específicas faltan para PyQt6
"""

import os
import sys
import ctypes
from pathlib import Path

def check_dll(dll_name):
    """Verificar si una DLL está disponible"""
    try:
        ctypes.CDLL(dll_name)
        return True
    except OSError:
        return False

def find_dll_in_system(dll_name):
    """Buscar DLL en directorios del sistema"""
    search_paths = [
        os.environ.get('SystemRoot', 'C:\\Windows') + '\\System32',
        os.environ.get('SystemRoot', 'C:\\Windows') + '\\SysWOW64',
        os.environ.get('ProgramFiles', 'C:\\Program Files'),
        os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'),
    ]
    
    # También buscar en PATH
    path_dirs = os.environ.get('PATH', '').split(';')
    search_paths.extend(path_dirs)
    
    for search_path in search_paths:
        if not search_path:
            continue
        dll_path = Path(search_path) / dll_name
        if dll_path.exists():
            return str(dll_path)
    
    return None

def check_pyqt6_dlls():
    """Verificar DLLs necesarias para PyQt6"""
    
    print("🔍 VERIFICANDO DLLs NECESARIAS PARA PyQt6")
    print("="*70)
    
    # DLLs críticas de Visual C++ Redistributable
    vc_dlls = [
        'vcruntime140.dll',
        'vcruntime140_1.dll', 
        'msvcp140.dll',
        'msvcp140_1.dll',
        'msvcp140_2.dll',
        'concrt140.dll',
    ]
    
    print("\n📦 Visual C++ Redistributable DLLs:")
    print("-"*70)
    
    missing_dlls = []
    found_dlls = []
    
    for dll in vc_dlls:
        if check_dll(dll):
            location = find_dll_in_system(dll)
            print(f"   ✅ {dll:<25} → {location or 'Disponible en PATH'}")
            found_dlls.append(dll)
        else:
            print(f"   ❌ {dll:<25} → NO ENCONTRADA")
            missing_dlls.append(dll)
    
    # Verificar PyQt6 DLLs específicas
    print("\n📦 PyQt6 DLLs:")
    print("-"*70)
    
    try:
        import PyQt6
        pyqt6_path = Path(PyQt6.__file__).parent
        
        pyqt6_dlls = [
            'Qt6Core.dll',
            'Qt6Gui.dll',
            'Qt6Widgets.dll',
        ]
        
        for dll in pyqt6_dlls:
            dll_path = pyqt6_path / 'Qt6' / 'bin' / dll
            if dll_path.exists():
                print(f"   ✅ {dll:<25} → {dll_path}")
            else:
                print(f"   ❌ {dll:<25} → NO ENCONTRADA")
                missing_dlls.append(dll)
    
    except ImportError:
        print("   ⚠️  PyQt6 no está instalado o no se puede importar")
    
    # Verificar arquitectura del sistema
    print("\n💻 Información del Sistema:")
    print("-"*70)
    print(f"   Python: {sys.version}")
    print(f"   Arquitectura Python: {sys.maxsize > 2**32 and '64-bit' or '32-bit'}")
    
    import platform
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Arquitectura OS: {platform.machine()}")
    
    # Resumen
    print("\n" + "="*70)
    print("📊 RESUMEN")
    print("="*70)
    
    if not missing_dlls:
        print("\n✅ Todas las DLLs necesarias están presentes!")
        print("\nSi PyQt6 sigue sin funcionar, el problema puede ser:")
        print("   1. Versión incompatible de las DLLs")
        print("   2. Problema de permisos")
        print("   3. Conflicto con otras instalaciones")
        
        print("\n🔧 Intenta reinstalar PyQt6:")
        print("   python /mnt/user-data/outputs/fix_pyqt6.py")
    
    else:
        print(f"\n❌ Faltan {len(missing_dlls)} DLLs:")
        for dll in missing_dlls:
            print(f"   - {dll}")
        
        # Determinar qué paquete instalar
        vc_missing = [dll for dll in missing_dlls if dll.startswith('vc') or dll.startswith('msvc') or dll.startswith('concrt')]
        
        if vc_missing:
            print("\n💡 SOLUCIÓN:")
            print("\n1. Descargar Visual C++ Redistributable 2015-2022 (x64):")
            print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("\n2. Ejecutar el instalador")
            print("\n3. Reiniciar el sistema")
            print("\n4. Verificar nuevamente:")
            print("   python /mnt/user-data/outputs/check_dlls.py")
        
        print("\n📝 Nota: Asegúrate de descargar la versión x64 (64-bit)")
        print("         si tu Python es de 64-bit (como parece ser)")
    
    # Información adicional
    print("\n" + "="*70)
    print("ℹ️  INFORMACIÓN ADICIONAL")
    print("="*70)
    
    print("\nVisual C++ Redistributable incluye:")
    print("   - Runtime libraries para C++")
    print("   - Componentes necesarios para PyQt6")
    print("   - Compatible con Windows 7 SP1 o superior")
    
    print("\nAlternativas mientras instalas:")
    print("   1. Modo headless (sin GUI):")
    print("      python main.py --headless")
    print("\n   2. Usar tests y backtesting:")
    print("      python test_phase3.py")
    
    return len(missing_dlls) == 0

if __name__ == "__main__":
    success = check_pyqt6_dlls()
    sys.exit(0 if success else 1)