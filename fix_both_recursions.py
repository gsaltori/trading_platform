#!/usr/bin/env python3
"""
Arreglar AMBAS recursiones en ml_engine.py
"""

import os
import shutil
from datetime import datetime

def fix_both_recursions():
    """Arreglar ambas recursiones encontradas"""
    
    filepath = "ml/ml_engine.py"
    
    print(f"🔧 Reparando AMBAS recursiones en {filepath}...")
    
    # Crear backup
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"   ✓ Backup: {backup_path}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        fixed_count = 0
        
        # Revisar cada línea
        for i, line in enumerate(lines):
            # Si encontramos la recursión problemática
            if 'self.feature_engineer = FeatureEngineer()' in line:
                indent = len(line) - len(line.lstrip())
                spaces = ' ' * indent
                
                # Determinar qué clase es mirando hacia atrás
                class_name = None
                for j in range(i-1, max(0, i-20), -1):
                    if 'class FeatureEngineer' in lines[j]:
                        class_name = 'FeatureEngineer'
                        break
                    elif 'class MLEngine' in lines[j]:
                        class_name = 'MLEngine'
                        break
                
                print(f"   📍 Línea {i+1} en clase {class_name}: {line.strip()}")
                
                if class_name == 'FeatureEngineer':
                    # Arreglar FeatureEngineer.__init__
                    lines[i] = f"{spaces}self.scaler = StandardScaler()\n"
                    # Agregar las otras líneas que faltan
                    lines.insert(i+1, f"{spaces}self.feature_selector = None\n")
                    lines.insert(i+2, f"{spaces}self.fitted = False\n")
                    print(f"   ✅ Arreglada en FeatureEngineer")
                    fixed_count += 1
                    
                elif class_name == 'MLEngine':
                    # Esta es correcta, NO cambiar
                    print(f"   ✓ Esta es correcta (MLEngine debe tener FeatureEngineer)")
                else:
                    print(f"   ⚠️  No se pudo determinar la clase")
        
        # Guardar
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"✅ Recursiones arregladas: {fixed_count}")
        return fixed_count > 0
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_fix():
    """Verificar que el fix funciona"""
    print("\n🧪 Verificando fix...")
    
    import sys
    # Limpiar cache
    for module in list(sys.modules.keys()):
        if module.startswith('ml'):
            del sys.modules[module]
    
    try:
        from ml.ml_engine import MLEngine, FeatureEngineer
        
        # Crear FeatureEngineer
        print("   Creando FeatureEngineer...")
        fe = FeatureEngineer()
        print(f"   ✅ FeatureEngineer OK (tiene scaler: {hasattr(fe, 'scaler')})")
        
        # Crear MLEngine
        print("   Creando MLEngine...")
        ml = MLEngine()
        print(f"   ✅ MLEngine OK (tiene feature_engineer: {hasattr(ml, 'feature_engineer')})")
        
        return True
        
    except RecursionError as e:
        print(f"   ❌ Todavía hay recursión!")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🔧 ARREGLANDO RECURSIÓN INFINITA")
    print("=" * 60)
    print()
    
    if fix_both_recursions():
        if verify_fix():
            print()
            print("=" * 60)
            print("✅ ¡TODO ARREGLADO!")
            print("=" * 60)
            print()
            print("Ahora ejecuta:")
            print("  del /s /q ml\\__pycache__")
            print("  python -m pytest tests/test_suite.py -v")
            return 0
        else:
            print()
            print("⚠️  Arreglado pero hay problemas")
            return 1
    else:
        print()
        print("❌ No se pudo arreglar")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())