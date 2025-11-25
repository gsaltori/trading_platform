#!/usr/bin/env python3
"""
Convertir imports de PyQt6 a PyQt5 en ui/main_window.py
"""

def fix_pyqt5_imports():
    """Arreglar imports para PyQt5"""
    
    file_path = 'ui/main_window.py'
    
    print("🔧 Convirtiendo imports de PyQt6 a PyQt5...")
    print("="*70)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Cambio 1: PyQt6 -> PyQt5
        content = content.replace('from PyQt6.QtWidgets', 'from PyQt5.QtWidgets')
        content = content.replace('from PyQt6.QtCore', 'from PyQt5.QtCore')
        content = content.replace('from PyQt6.QtGui', 'from PyQt5.QtGui')
        content = content.replace('from PyQt6.QtCharts', 'from PyQt5.QtChart')
        content = content.replace('from PyQt6 import', 'from PyQt5 import')
        
        print("✅ Cambiados imports PyQt6 -> PyQt5")
        
        # Cambio 2: QAction está en QtWidgets en PyQt5, no en QtGui
        old_import = "from PyQt5.QtGui import QAction, QIcon, QFont"
        new_import = "from PyQt5.QtWidgets import QAction\nfrom PyQt5.QtGui import QIcon, QFont"
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            print("✅ Movido QAction de QtGui a QtWidgets")
        
        # Cambio 3: pyqtSignal está en QtCore
        content = content.replace('from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal',
                                 'from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal')
        
        # Cambio 4: Qt.AlignmentFlag -> Qt en PyQt5
        content = content.replace('Qt.AlignmentFlag.Align', 'Qt.Align')
        content = content.replace('QFont.Weight.Bold', 'QFont.Bold')
        
        print("✅ Ajustados flags de Qt para PyQt5")
        
        # Cambio 5: QtCharts (diferente en PyQt5)
        if 'QtChart' in content:
            print("⚠️  QtChart detectado - puede requerir ajustes manuales")
        
        # Escribir cambios
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("\n" + "="*70)
        print("✅ ARCHIVO ACTUALIZADO EXITOSAMENTE")
        print("="*70)
        print(f"\nArchivo: {file_path}")
        print("\nCambios aplicados:")
        print("  1. ✅ PyQt6 -> PyQt5")
        print("  2. ✅ QAction movido a QtWidgets")
        print("  3. ✅ Flags de Qt ajustados")
        print("  4. ✅ Compatibilidad con PyQt5")
        
        print("\n🎉 ¡Listo! Ahora ejecuta:")
        print("   python main.py --gui")
        
        return True
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: No se encontró el archivo {file_path}")
        print(f"\nAsegúrate de ejecutar este script desde la raíz del proyecto")
        return False
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_pyqt5_imports()
    
    if success:
        print("\n" + "="*70)
        print("📋 VERIFICACIÓN")
        print("="*70)
        print("\nPara verificar que funciona:")
        print("  python -c \"from PyQt5.QtWidgets import QApplication; print('OK')\"")
        print("\nLuego ejecuta la GUI:")
        print("  python main.py --gui")
    else:
        print("\n💡 ALTERNATIVA: Usa la GUI simple con Tkinter")
        print("  python /mnt/user-data/outputs/simple_gui.py")