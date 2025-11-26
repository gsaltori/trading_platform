#!/usr/bin/env python3
"""
DETECTOR DE INSTALACIONES MT5

Busca automáticamente todas las instalaciones de MetaTrader 5 en el sistema.
"""

import os
import winreg
from pathlib import Path

def find_mt5_installations():
    """Buscar todas las instalaciones MT5 en el sistema"""
    
    installations = []
    
    print("🔍 Buscando instalaciones de MetaTrader 5...")
    print("="*70)
    
    # 1. Buscar en registro de Windows
    print("\n1️⃣ Buscando en el Registro de Windows...")
    
    registry_paths = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]
    
    for hkey, subkey_path in registry_paths:
        try:
            key = winreg.OpenKey(hkey, subkey_path)
            for i in range(winreg.QueryInfoKey(key)[0]):
                try:
                    subkey_name = winreg.EnumKey(key, i)
                    subkey = winreg.OpenKey(key, subkey_name)
                    
                    try:
                        display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                        if "metatrader 5" in display_name.lower():
                            try:
                                install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                                terminal_path = Path(install_location) / "terminal64.exe"
                                
                                if terminal_path.exists():
                                    installations.append({
                                        'name': display_name,
                                        'path': str(terminal_path),
                                        'source': 'Registry'
                                    })
                                    print(f"   ✓ Encontrado: {display_name}")
                                    print(f"     Ruta: {terminal_path}")
                            except:
                                pass
                    except:
                        pass
                    
                    winreg.CloseKey(subkey)
                except:
                    pass
            winreg.CloseKey(key)
        except:
            pass
    
    # 2. Buscar en directorios comunes
    print("\n2️⃣ Buscando en directorios comunes...")
    
    common_dirs = [
        Path("C:/Program Files/MetaTrader 5"),
        Path("C:/Program Files (x86)/MetaTrader 5"),
        Path(os.path.expanduser("~")) / "AppData/Roaming/MetaQuotes/Terminal",
    ]
    
    # Buscar también variantes con nombre de broker
    program_files = Path("C:/Program Files")
    if program_files.exists():
        for item in program_files.iterdir():
            if item.is_dir() and "metatrader" in item.name.lower():
                common_dirs.append(item)
    
    for directory in common_dirs:
        if directory.exists():
            # Buscar terminal64.exe
            for terminal_path in directory.rglob("terminal64.exe"):
                # Evitar duplicados
                path_str = str(terminal_path)
                if not any(inst['path'] == path_str for inst in installations):
                    installations.append({
                        'name': f"MT5 - {terminal_path.parent.name}",
                        'path': path_str,
                        'source': 'File Search'
                    })
                    print(f"   ✓ Encontrado: {terminal_path.parent.name}")
                    print(f"     Ruta: {terminal_path}")
    
    # 3. Buscar en todas las unidades (solo raíz y carpetas principales)
    print("\n3️⃣ Búsqueda rápida en otras unidades...")
    
    for drive in ['D:', 'E:', 'F:']:
        drive_path = Path(drive + '/')
        if drive_path.exists():
            # Buscar solo en el primer nivel
            for item in drive_path.iterdir():
                if item.is_dir() and any(keyword in item.name.lower() for keyword in ['mt5', 'metatrader', 'trading']):
                    for terminal_path in item.rglob("terminal64.exe"):
                        path_str = str(terminal_path)
                        if not any(inst['path'] == path_str for inst in installations):
                            installations.append({
                                'name': f"MT5 - {terminal_path.parent.name}",
                                'path': path_str,
                                'source': f'Drive {drive}'
                            })
                            print(f"   ✓ Encontrado en {drive}: {terminal_path.parent.name}")
    
    return installations

def main():
    installations = find_mt5_installations()
    
    print("\n" + "="*70)
    print("📊 RESUMEN")
    print("="*70)
    
    if installations:
        print(f"\n✅ Se encontraron {len(installations)} instalación(es) de MT5:\n")
        
        for i, inst in enumerate(installations, 1):
            print(f"{i}. {inst['name']}")
            print(f"   Ruta: {inst['path']}")
            print(f"   Fuente: {inst['source']}")
            print()
        
        print("="*70)
        print("💡 CÓMO USAR EN LA GUI")
        print("="*70)
        
        print("\nEn la GUI mejorada:")
        print("1. Ve a la pestaña '📊 Dashboard'")
        print("2. En 'Instalación MT5' pega una de estas rutas:")
        print()
        
        for i, inst in enumerate(installations, 1):
            print(f"   Opción {i}: {inst['path']}")
        
        print("\n3. O usa el botón '📁 Buscar' para navegar manualmente")
        print("4. Haz clic en '🔌 Conectar MT5'")
        
        # Guardar a archivo JSON
        import json
        output_file = Path("mt5_installations.json")
        
        with open(output_file, 'w') as f:
            json.dump(installations, f, indent=2)
        
        print(f"\n✅ Información guardada en: {output_file}")
        
    else:
        print("\n❌ No se encontraron instalaciones de MT5")
        print("\nVerifica que MetaTrader 5 esté instalado en tu sistema.")
        print("\nSi tienes MT5 instalado en una ubicación no estándar,")
        print("usa el botón '📁 Buscar' en la GUI para seleccionarlo manualmente.")
    
    print("\n" + "="*70)
    print("🔧 INSTALACIONES PORTABLES")
    print("="*70)
    
    print("\nSi usas una instalación portable (sin instalar):")
    print("1. Busca la carpeta donde descomprimiste MT5")
    print("2. Dentro debe haber un archivo 'terminal64.exe'")
    print("3. Copia la ruta completa, por ejemplo:")
    print("   D:/MisAplicaciones/MT5Portable/terminal64.exe")
    print("4. Pégala en el campo 'Instalación MT5' de la GUI")
    
    print("\n" + "="*70)
    
    input("\nPresiona Enter para salir...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("\nPresiona Enter para salir...")