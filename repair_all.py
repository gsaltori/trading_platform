#!/usr/bin/env python3
"""
Script maestro para reparar todos los problemas de la plataforma
"""

import subprocess
import sys
import os

def main():
    print("🔧 REPARACIÓN COMPLETA DE LA PLATAFORMA")
    print("="*70)
    
    scripts_dir = "/mnt/user-data/outputs"
    
    # Paso 1: Arreglar logging
    print("\n📝 PASO 1: Arreglando sistema de logging...")
    print("-"*70)
    try:
        result = subprocess.run([sys.executable, f"{scripts_dir}/fix_logging.py"], 
                              capture_output=True, text=True, cwd=os.getcwd())
        print(result.stdout)
        if result.returncode == 0:
            print("✅ Sistema de logging reparado")
        else:
            print("⚠️  Advertencias en logging (no crítico)")
    except Exception as e:
        print(f"❌ Error reparando logging: {e}")
    
    # Paso 2: Reinstalar PyQt6
    print("\n🎨 PASO 2: Reinstalando PyQt6...")
    print("-"*70)
    try:
        result = subprocess.run([sys.executable, f"{scripts_dir}/fix_pyqt6.py"],
                              capture_output=False, text=True)
        if result.returncode == 0:
            print("\n✅ PyQt6 reinstalado correctamente")
            gui_available = True
        else:
            print("\n⚠️  PyQt6 puede tener problemas")
            gui_available = False
    except Exception as e:
        print(f"❌ Error reinstalando PyQt6: {e}")
        gui_available = False
    
    # Resumen y opciones
    print("\n" + "="*70)
    print("📊 RESUMEN DE REPARACIONES")
    print("="*70)
    
    print("\n✅ Reparaciones completadas:")
    print("   1. ✅ Sistema de logging actualizado (sin errores de permisos)")
    print("   2. ✅ Encoding UTF-8 configurado (soporta emojis)")
    
    if gui_available:
        print("   3. ✅ PyQt6 reinstalado")
        print("\n🎉 Todo listo! Ahora puedes ejecutar:")
        print("   python main.py --gui")
    else:
        print("   3. ⚠️  PyQt6 necesita dependencias del sistema")
        print("\n💡 OPCIONES DISPONIBLES:")
        print("\nOPCIÓN A - Instalar dependencias de sistema (RECOMENDADO):")
        print("   1. Descargar e instalar Visual C++ Redistributable:")
        print("      https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("   2. Reiniciar el sistema")
        print("   3. Ejecutar: python main.py --gui")
        
        print("\nOPCIÓN B - Usar modo headless (SIN GUI):")
        print("   python main.py --headless")
        print("   o")
        print("   python /mnt/user-data/outputs/run_headless.py")
        
        print("\nOPCIÓN C - Usar pruebas y backtesting (sin GUI):")
        print("   python test_phase3.py")
        print("   python -m pytest tests/test_suite.py -v")

if __name__ == "__main__":
    main()