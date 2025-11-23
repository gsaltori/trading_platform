#!/usr/bin/env python3
"""
Diagnóstico profundo de ml_engine.py
"""

import sys
import os
import traceback

def deep_diagnosis():
    """Diagnóstico profundo del error"""
    print("=" * 60)
    print("🔍 DIAGNÓSTICO PROFUNDO")
    print("=" * 60)
    
    # Agregar directorio al path
    if '.' not in sys.path:
        sys.path.insert(0, '.')
    
    print("\n1️⃣ Intentando importar el módulo completo...")
    try:
        import ml.ml_engine as ml_module
        print("✅ Módulo importado exitosamente")
        
        # Listar lo que hay en el módulo
        print("\n📦 Contenido del módulo:")
        items = dir(ml_module)
        classes = [item for item in items if item[0].isupper() and not item.startswith('_')]
        functions = [item for item in items if item[0].islower() and not item.startswith('_')]
        
        print(f"\n   Clases encontradas ({len(classes)}):")
        for cls in classes:
            print(f"      - {cls}")
        
        print(f"\n   Funciones encontradas ({len(functions)}):")
        for func in functions[:10]:  # Solo primeras 10
            print(f"      - {func}")
        
        # Verificar clases esperadas
        print("\n2️⃣ Verificando clases esperadas...")
        expected = ['MLEngine', 'MLModelConfig', 'FeatureEngineer', 'MarketRegimeDetector']
        missing = []
        
        for cls_name in expected:
            if hasattr(ml_module, cls_name):
                print(f"   ✅ {cls_name}")
            else:
                print(f"   ❌ {cls_name} NO ENCONTRADA")
                missing.append(cls_name)
        
        if missing:
            print(f"\n⚠️  Clases faltantes: {', '.join(missing)}")
            print("\n💡 Posible causa: Error en la definición de estas clases")
            return False
        else:
            print("\n✅ Todas las clases encontradas correctamente")
            return True
            
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("\n📋 Traceback completo:")
        traceback.print_exc()
        
        print("\n3️⃣ Intentando importación paso a paso...")
        try:
            print("   a) Importando pandas...")
            import pandas as pd
            print("      ✅ pandas OK")
            
            print("   b) Importando numpy...")
            import numpy as np
            print("      ✅ numpy OK")
            
            print("   c) Importando sklearn...")
            from sklearn.ensemble import RandomForestClassifier
            print("      ✅ sklearn OK")
            
            print("   d) Importando tensorflow...")
            import tensorflow as tf
            print("      ✅ tensorflow OK")
            
            print("   e) Importando ta (indicadores técnicos)...")
            try:
                import talib as ta
                print("      ✅ talib OK")
            except ImportError:
                print("      ⚠️  talib no disponible, intentando ta...")
                try:
                    import ta
                    print("      ✅ ta OK")
                except ImportError:
                    print("      ❌ ta no disponible")
            
            print("\n   Todas las dependencias básicas están OK")
            print("   El error está en el código de ml_engine.py")
            
        except Exception as dep_error:
            print(f"\n❌ Error en dependencia: {dep_error}")
            return False
        
        # Intentar ejecutar el archivo directamente
        print("\n4️⃣ Intentando ejecutar el archivo directamente...")
        try:
            with open('ml/ml_engine.py', 'r', encoding='utf-8') as f:
                code = f.read()
            
            exec(compile(code, 'ml/ml_engine.py', 'exec'))
            print("✅ Ejecución directa exitosa")
        except Exception as exec_error:
            print(f"❌ Error en ejecución: {exec_error}")
            print("\n📋 Traceback:")
            traceback.print_exc()
            
            # Intentar encontrar la línea exacta
            print("\n5️⃣ Buscando línea del error...")
            import re
            tb = traceback.format_exc()
            
            # Buscar líneas que mencionen ml_engine.py
            lines = tb.split('\n')
            for line in lines:
                if 'ml_engine.py' in line and 'line' in line.lower():
                    print(f"   📍 {line}")
        
        return False
    
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        print("\n📋 Traceback completo:")
        traceback.print_exc()
        return False

def check_class_definitions():
    """Verificar definiciones de clases en el archivo"""
    print("\n" + "=" * 60)
    print("🔍 VERIFICANDO DEFINICIONES DE CLASES")
    print("=" * 60)
    
    try:
        with open('ml/ml_engine.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar definiciones de clases
        import re
        class_pattern = r'^class\s+(\w+)'
        
        classes_found = []
        for i, line in enumerate(content.split('\n'), 1):
            match = re.match(class_pattern, line)
            if match:
                class_name = match.group(1)
                classes_found.append((class_name, i))
                print(f"   Línea {i:4d}: class {class_name}")
        
        expected = ['MLModelConfig', 'MLResult', 'FeatureEngineer', 'MLEngine', 'MarketRegimeDetector']
        
        print(f"\n📊 Resumen:")
        print(f"   Clases encontradas: {len(classes_found)}")
        print(f"   Clases esperadas: {len(expected)}")
        
        found_names = [name for name, _ in classes_found]
        missing = [name for name in expected if name not in found_names]
        
        if missing:
            print(f"\n⚠️  Clases faltantes: {', '.join(missing)}")
        else:
            print(f"\n✅ Todas las clases esperadas están definidas")
        
        return len(missing) == 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    success = deep_diagnosis()
    check_class_definitions()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ DIAGNÓSTICO EXITOSO")
        print("\nEl módulo se puede importar correctamente.")
        print("El error en pytest puede ser por otro motivo.")
    else:
        print("❌ PROBLEMA ENCONTRADO")
        print("\n💡 SOLUCIONES:")
        print("\n1. Si el error está en una línea específica:")
        print("   - Abre ml/ml_engine.py")
        print("   - Ve a la línea indicada arriba")
        print("   - Corrige el error (usualmente un import faltante)")
        print("\n2. Si falta una clase:")
        print("   - Verifica que la clase esté definida en el archivo")
        print("   - Verifica la indentación")
        print("\n3. Si es un error de dependencia:")
        print("   - Instala la dependencia faltante con pip")
        print("\n4. Solución rápida:")
        print("   python repair_ml_engine.py")
    
    print("=" * 60)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())