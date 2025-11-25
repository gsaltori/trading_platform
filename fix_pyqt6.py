#!/usr/bin/env python3
"""
Script para reinstalar PyQt6 y solucionar problemas de DLL
"""

import subprocess
import sys

def reinstall_pyqt6():
    """Reinstalar PyQt6 completamente"""
    
    print("🔧 Reparando instalación de PyQt6...")
    print("\n" + "="*60)
    
    steps = [
        ("Desinstalando PyQt6...", [sys.executable, "-m", "pip", "uninstall", "-y", "PyQt6", "PyQt6-Qt6", "PyQt6-sip"]),
        ("Desinstalando PyQt6-WebEngine...", [sys.executable, "-m", "pip", "uninstall", "-y", "PyQt6-WebEngine"]),
        ("Limpiando cache de pip...", [sys.executable, "-m", "pip", "cache", "purge"]),
        ("Instalando PyQt6...", [sys.executable, "-m", "pip", "install", "--no-cache-dir", "PyQt6==6.6.1"]),
        ("Instalando PyQt6-WebEngine...", [sys.executable, "-m", "pip", "install", "--no-cache-dir", "PyQt6-WebEngine==6.6.0"]),
    ]
    
    for description, command in steps:
        print(f"\n{description}")
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {description} - Completado")
            else:
                print(f"⚠️  {description} - Con advertencias")
                if result.stderr:
                    print(f"   {result.stderr[:200]}")
        except Exception as e:
            print(f"❌ {description} - Error: {e}")
    
    print("\n" + "="*60)
    print("\n🧪 Verificando instalación...")
    
    # Verificar importación
    try:
        subprocess.run([sys.executable, "-c", "from PyQt6.QtWidgets import QApplication; print('✅ PyQt6 funciona correctamente')"], check=True)
        print("\n🎉 ¡PyQt6 instalado y funcionando correctamente!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ PyQt6 sigue con problemas")
        print("\n💡 Soluciones alternativas:")
        print("   1. Instalar Microsoft Visual C++ Redistributable:")
        print("      https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("   2. Reiniciar el sistema")
        print("   3. Ejecutar en modo headless: python main.py --headless")
        return False

if __name__ == "__main__":
    reinstall_pyqt6()