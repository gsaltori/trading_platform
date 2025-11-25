#!/usr/bin/env python3
"""
Verificar y arreglar MLEngine.__init__ y FeatureEngineer.__init__
"""

import shutil
from datetime import datetime

filepath = "ml/ml_engine.py"

print("🔍 Verificando archivo...")
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Buscar las dos clases y sus __init__
feature_engineer_init_line = None
mlengine_init_line = None

for i, line in enumerate(lines):
    if 'class FeatureEngineer:' in line:
        # Buscar su __init__
        for j in range(i+1, min(i+10, len(lines))):
            if 'def __init__(self):' in lines[j]:
                feature_engineer_init_line = j
                break
    elif 'class MLEngine:' in line:
        # Buscar su __init__
        for j in range(i+1, min(i+10, len(lines))):
            if 'def __init__(self):' in lines[j]:
                mlengine_init_line = j
                break

print(f"\n📊 Estado actual:")
print(f"   FeatureEngineer.__init__ en línea: {feature_engineer_init_line + 1 if feature_engineer_init_line else 'NO ENCONTRADO'}")
print(f"   MLEngine.__init__ en línea: {mlengine_init_line + 1 if mlengine_init_line else 'NO ENCONTRADO'}")

# Ver qué tienen
if feature_engineer_init_line:
    print(f"\n   FeatureEngineer.__init__ contiene:")
    for j in range(feature_engineer_init_line, min(feature_engineer_init_line + 5, len(lines))):
        print(f"      {j+1}: {lines[j].rstrip()}")

if mlengine_init_line:
    print(f"\n   MLEngine.__init__ contiene:")
    for j in range(mlengine_init_line, min(mlengine_init_line + 5, len(lines))):
        print(f"      {j+1}: {lines[j].rstrip()}")

# Backup
backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(filepath, backup_path)
print(f"\n✓ Backup: {backup_path}")

# Arreglar FeatureEngineer.__init__
if feature_engineer_init_line:
    next_line = feature_engineer_init_line + 1
    content = lines[next_line].strip()
    
    if 'self.scaler' not in content and 'self.feature_engineer' in content:
        print(f"\n🔧 Arreglando FeatureEngineer.__init__...")
        indent = '        '
        # Reemplazar la línea problemática y agregar las correctas
        lines[next_line] = f"{indent}self.scaler = StandardScaler()\n"
        lines.insert(next_line + 1, f"{indent}self.feature_selector = None\n")
        lines.insert(next_line + 2, f"{indent}self.fitted = False\n")
        print("   ✅ FeatureEngineer.__init__ arreglado")

# Verificar/Arreglar MLEngine.__init__
if mlengine_init_line:
    next_line = mlengine_init_line + 1
    content = lines[next_line].strip()
    
    if 'self.feature_engineer' not in content:
        print(f"\n🔧 Arreglando MLEngine.__init__...")
        indent = '        '
        # Agregar la inicialización correcta
        lines.insert(next_line, f"{indent}self.feature_engineer = FeatureEngineer()\n")
        lines.insert(next_line + 1, f"{indent}self.models = {{}}\n")
        lines.insert(next_line + 2, f"{indent}self.results = {{}}\n")
        print("   ✅ MLEngine.__init__ arreglado")
    else:
        print(f"\n   ✓ MLEngine.__init__ parece OK")

# Guardar
with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("\n🧪 Verificando...")
import sys
for module in list(sys.modules.keys()):
    if module.startswith('ml'):
        del sys.modules[module]

try:
    from ml.ml_engine import MLEngine, FeatureEngineer
    
    print("   Creando FeatureEngineer...")
    fe = FeatureEngineer()
    assert hasattr(fe, 'scaler'), "FeatureEngineer no tiene scaler"
    assert hasattr(fe, 'fitted'), "FeatureEngineer no tiene fitted"
    print(f"   ✅ FeatureEngineer OK")
    
    print("   Creando MLEngine...")
    ml = MLEngine()
    assert hasattr(ml, 'feature_engineer'), "MLEngine no tiene feature_engineer"
    assert hasattr(ml, 'models'), "MLEngine no tiene models"
    assert hasattr(ml, 'results'), "MLEngine no tiene results"
    print(f"   ✅ MLEngine OK")
    
    print("\n" + "="*60)
    print("✅ ¡TODO FUNCIONANDO!")
    print("="*60)
    print("\nEjecuta los tests:")
    print("  python -m pytest tests/test_suite.py::TestTradingPlatform::test_05_machine_learning_engine -v")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n   Restaurando backup...")
    shutil.copy2(backup_path, filepath)