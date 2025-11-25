#!/usr/bin/env python3
"""
Reparar recursión infinita en FeatureEngineer
"""

import os
import shutil
from datetime import datetime

def fix_recursion():
    """Arreglar la recursión infinita en FeatureEngineer.__init__"""
    
    filepath = "ml/ml_engine.py"
    
    if not os.path.exists(filepath):
        print(f"❌ Archivo no encontrado: {filepath}")
        return False
    
    print(f"🔧 Reparando recursión infinita en {filepath}...")
    
    # Crear backup
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"   ✓ Backup: {backup_path}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # El problema está en la línea 236
        # FeatureEngineer.__init__ tiene: self.feature_engineer = FeatureEngineer()
        # Debe tener: self.scaler = StandardScaler()
        
        # Buscar y reemplazar SOLO en FeatureEngineer.__init__
        lines = content.split('\n')
        fixed_lines = []
        in_feature_engineer_init = False
        fixed_count = 0
        
        for i, line in enumerate(lines):
            # Detectar si estamos en FeatureEngineer.__init__
            if 'class FeatureEngineer:' in line:
                in_feature_engineer_init = False
            elif in_feature_engineer_init and (line.strip().startswith('def ') or line.strip().startswith('class ')):
                in_feature_engineer_init = False
            elif 'def __init__(self):' in line:
                # Verificar si estamos justo después de FeatureEngineer
                # Buscamos hacia atrás para confirmarlo
                for j in range(i-1, max(0, i-10), -1):
                    if 'class FeatureEngineer' in lines[j]:
                        in_feature_engineer_init = True
                        break
                    elif 'class ' in lines[j]:
                        break
            
            # Arreglar la línea problemática
            if in_feature_engineer_init and 'self.feature_engineer = FeatureEngineer()' in line:
                # Reemplazar con la inicialización correcta
                indent = len(line) - len(line.lstrip())
                fixed_lines.append(' ' * indent + 'self.scaler = StandardScaler()')
                fixed_lines.append(' ' * indent + 'self.feature_selector = None')
                fixed_lines.append(' ' * indent + 'self.fitted = False')
                fixed_count += 1
                print(f"   ✓ Línea {i+1}: Arreglada recursión infinita")
            else:
                fixed_lines.append(line)
        
        if fixed_count == 0:
            print("   ⚠️  No se encontró la recursión para arreglar")
            print("   Buscando manualmente...")
            
            # Buscar todas las ocurrencias
            for i, line in enumerate(lines, 1):
                if 'self.feature_engineer = FeatureEngineer()' in line:
                    print(f"   📍 Encontrada en línea {i}: {line.strip()}")
            
            return False
        
        # Guardar archivo
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines))
        
        print(f"✅ Recursión arreglada ({fixed_count} ocurrencia(s))")
        return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_fix():
    """Verificar que el archivo está OK"""
    print("\n🧪 Verificando fix...")
    
    try:
        # Limpiar cache
        import sys
        if 'ml.ml_engine' in sys.modules:
            del sys.modules['ml.ml_engine']
        if 'ml' in sys.modules:
            del sys.modules['ml']
        
        # Intentar importar
        from ml.ml_engine import MLEngine, FeatureEngineer
        
        # Intentar crear instancia
        fe = FeatureEngineer()
        print("✅ FeatureEngineer se crea correctamente")
        
        ml = MLEngine()
        print("✅ MLEngine se crea correctamente")
        
        return True
        
    except RecursionError as e:
        print(f"❌ Todavía hay recursión: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🔧 REPARANDO RECURSIÓN INFINITA")
    print("=" * 60)
    print()
    
    if fix_recursion():
        if verify_fix():
            print()
            print("=" * 60)
            print("✅ ¡RECURSIÓN ARREGLADA!")
            print("=" * 60)
            print()
            print("Limpiar cache y ejecutar tests:")
            print("  del /s /q ml\\__pycache__")
            print("  python -m pytest tests/test_suite.py::TestTradingPlatform::test_05_machine_learning_engine -v")
            return 0
        else:
            print()
            print("⚠️  Arreglado pero hay problemas verificando")
            return 1
    else:
        print()
        print("❌ No se pudo arreglar")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())