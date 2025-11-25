#!/usr/bin/env python3
"""
Reemplazo ultra directo de la recursión
"""

import shutil
from datetime import datetime

filepath = "ml/ml_engine.py"

print("🔧 Abriendo archivo...")
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Backup
backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(filepath, backup_path)
print(f"✓ Backup: {backup_path}")

# Contar ocurrencias
count = content.count('self.feature_engineer = FeatureEngineer()')
print(f"\n📊 Encontradas {count} ocurrencias de 'self.feature_engineer = FeatureEngineer()'")

# Estrategia: reemplazar SOLO LA PRIMERA ocurrencia
# (que es la problemática en FeatureEngineer)

parts = content.split('self.feature_engineer = FeatureEngineer()', 1)

if len(parts) == 2:
    # Reemplazar la primera ocurrencia con la inicialización correcta
    new_content = parts[0] + '''self.scaler = StandardScaler()
        self.feature_selector = None
        self.fitted = False''' + parts[1]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Primera ocurrencia reemplazada")
    print("\n🧪 Verificando...")
    
    # Verificar
    import sys
    for module in list(sys.modules.keys()):
        if module.startswith('ml'):
            del sys.modules[module]
    
    try:
        from ml.ml_engine import MLEngine, FeatureEngineer
        
        print("   Creando FeatureEngineer...")
        fe = FeatureEngineer()
        print(f"   ✅ FeatureEngineer OK")
        
        print("   Creando MLEngine...")
        ml = MLEngine()
        print(f"   ✅ MLEngine OK")
        
        print("\n" + "="*60)
        print("✅ ¡TODO FUNCIONANDO!")
        print("="*60)
        print("\nEjecuta los tests:")
        print("  del /s /q ml\\__pycache__")
        print("  python -m pytest tests/test_suite.py -v")
        
    except RecursionError:
        print("   ❌ Todavía hay recursión")
        print("\n   Restaurando backup...")
        shutil.copy2(backup_path, filepath)
        print("   Archivo restaurado")
    except Exception as e:
        print(f"   ❌ Error: {e}")

else:
    print("❌ No se encontró la línea para reemplazar")