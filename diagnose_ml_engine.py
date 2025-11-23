#!/usr/bin/env python3
"""
Script de diagnóstico para encontrar errores en ml_engine.py
"""

import sys
import os

def check_syntax(filepath):
    """Verificar sintaxis de un archivo Python"""
    print(f"🔍 Verificando sintaxis de {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"❌ Archivo no encontrado: {filepath}")
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        compile(code, filepath, 'exec')
        print(f"✅ Sintaxis correcta en {filepath}")
        return True
    except SyntaxError as e:
        print(f"❌ Error de sintaxis en {filepath}:")
        print(f"   Línea {e.lineno}: {e.msg}")
        print(f"   Texto: {e.text}")
        print(f"   Posición: {' ' * (e.offset - 1) if e.offset else ''}^")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def check_imports(filepath):
    """Verificar que se pueden importar las clases principales"""
    print(f"\n🔍 Verificando imports de {filepath}...")
    
    # Agregar directorio al path
    module_dir = os.path.dirname(os.path.abspath(filepath))
    parent_dir = os.path.dirname(module_dir)
    
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        # Intentar importar el módulo
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        package_name = os.path.basename(module_dir)
        
        import importlib
        module = importlib.import_module(f'{package_name}.{module_name}')
        
        # Verificar clases esperadas
        expected_classes = ['MLEngine', 'MLModelConfig', 'FeatureEngineer', 'MarketRegimeDetector']
        
        for class_name in expected_classes:
            if hasattr(module, class_name):
                print(f"   ✅ {class_name} encontrada")
            else:
                print(f"   ❌ {class_name} NO encontrada")
        
        print(f"✅ Imports exitosos")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    print("=" * 60)
    print("🔧 DIAGNÓSTICO DE ml_engine.py")
    print("=" * 60)
    
    filepath = "ml/ml_engine.py"
    
    # Verificar sintaxis
    syntax_ok = check_syntax(filepath)
    
    # Verificar imports
    if syntax_ok:
        imports_ok = check_imports(filepath)
    else:
        imports_ok = False
    
    print("\n" + "=" * 60)
    if syntax_ok and imports_ok:
        print("✅ TODO OK - El archivo no tiene errores detectables")
    else:
        print("❌ ERRORES ENCONTRADOS - Revisa los mensajes arriba")
        
        if not syntax_ok:
            print("\n💡 Solución sugerida:")
            print("   1. Abre ml/ml_engine.py")
            print("   2. Ve a la línea indicada arriba")
            print("   3. Corrige el error de sintaxis")
            print("\n   O restaura el backup:")
            print("   cp ml/ml_engine.py.backup.* ml/ml_engine.py")
    
    return 0 if (syntax_ok and imports_ok) else 1

if __name__ == "__main__":
    sys.exit(main())