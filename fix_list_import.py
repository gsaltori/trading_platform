#!/usr/bin/env python3
"""
Fix exacto para ml_engine.py - Agrega List al import de typing
"""

import os
import shutil
from datetime import datetime

def fix_list_import():
    """Agregar List al import de typing"""
    
    filepath = "ml/ml_engine.py"
    
    if not os.path.exists(filepath):
        print(f"❌ Archivo no encontrado: {filepath}")
        return False
    
    print(f"🔧 Corrigiendo imports en {filepath}...")
    
    # Crear backup
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"   ✓ Backup: {backup_path}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Buscar la línea de import de typing
        fixed = False
        for i, line in enumerate(lines):
            if line.strip().startswith('from typing import') and 'List' not in line:
                # Agregar List al import
                old_line = line.rstrip()
                
                # Dividir los imports
                imports_part = old_line.split('from typing import')[1].strip()
                
                # Si ya tiene Tuple, agregamos List también
                if 'Tuple' in imports_part:
                    # Ya tiene otros imports, agregar List
                    if not imports_part.endswith(','):
                        new_line = old_line + ', List\n'
                    else:
                        new_line = old_line + ' List\n'
                else:
                    # No tiene nada útil, reemplazar completamente
                    new_line = 'from typing import Dict, List, Optional, Tuple, Any, Callable, Union\n'
                
                lines[i] = new_line
                print(f"   ✓ Línea {i+1} corregida:")
                print(f"      Antes: {old_line}")
                print(f"      Después: {new_line.rstrip()}")
                fixed = True
                break
        
        if not fixed:
            # No encontramos el import, buscar dónde agregarlo
            print("   ⚠️  No se encontró 'from typing import', agregando...")
            
            # Buscar después de los imports estándar
            insert_at = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    insert_at = i + 1
                elif line.strip() == '' and insert_at > 0:
                    break
            
            new_import = 'from typing import Dict, List, Optional, Tuple, Any, Callable, Union\n'
            lines.insert(insert_at, new_import)
            print(f"   ✓ Import agregado en línea {insert_at + 1}")
        
        # Guardar archivo
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"✅ Archivo corregido")
        
        # Verificar sintaxis
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        try:
            compile(code, filepath, 'exec')
            print(f"✅ Sintaxis correcta")
            return True
        except SyntaxError as e:
            print(f"❌ Error de sintaxis: {e}")
            print(f"   Restaurando backup...")
            shutil.copy2(backup_path, filepath)
            return False
        except NameError as e:
            print(f"⚠️  Error de nombre: {e}")
            print(f"   Esto es normal, se resolverá al importar")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_fix():
    """Verificar que el fix funcionó"""
    print("\n🧪 Verificando fix...")
    
    import sys
    if '.' not in sys.path:
        sys.path.insert(0, '.')
    
    try:
        # Limpiar cache de imports
        if 'ml.ml_engine' in sys.modules:
            del sys.modules['ml.ml_engine']
        if 'ml' in sys.modules:
            del sys.modules['ml']
        
        # Intentar importar
        from ml.ml_engine import MLEngine, MLModelConfig
        
        print("✅ Import exitoso!")
        print(f"   ✓ MLEngine: {MLEngine}")
        print(f"   ✓ MLModelConfig: {MLModelConfig}")
        return True
        
    except ImportError as e:
        print(f"❌ Todavía hay error de import: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🔧 FIX EXACTO PARA ml_engine.py")
    print("=" * 60)
    print()
    print("Problema detectado:")
    print("  ❌ Línea 34: name 'List' is not defined")
    print("  ❌ Falta: from typing import List")
    print()
    print("=" * 60)
    print()
    
    if fix_list_import():
        if verify_fix():
            print()
            print("=" * 60)
            print("✅ FIX EXITOSO!")
            print("=" * 60)
            print()
            print("Ahora ejecuta:")
            print("  python -m pytest tests/test_suite.py -v")
            return 0
        else:
            print()
            print("=" * 60)
            print("⚠️  FIX APLICADO PERO HAY OTROS ERRORES")
            print("=" * 60)
            print()
            print("El import de List se corrigió, pero hay otros problemas.")
            print("Ejecuta de nuevo:")
            print("  python deep_diagnose.py")
            return 1
    else:
        print()
        print("=" * 60)
        print("❌ FIX FALLÓ")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())