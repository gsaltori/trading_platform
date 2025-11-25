#!/usr/bin/env python3
"""
Encuentra TODOS los errores en ml_engine.py línea por línea
"""

import sys

def find_all_errors():
    """Ejecutar el archivo y encontrar todos los errores"""
    
    print("=" * 60)
    print("🔍 BUSCANDO TODOS LOS ERRORES")
    print("=" * 60)
    print()
    
    filepath = "ml/ml_engine.py"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        print("📋 Contenido del archivo cargado correctamente")
        print(f"   Líneas totales: {len(code.split(chr(10)))}")
        print()
        
        # Verificar imports de typing
        print("1️⃣ Verificando imports de typing...")
        lines = code.split('\n')
        
        typing_line = None
        for i, line in enumerate(lines[:30], 1):  # Primeras 30 líneas
            if 'from typing import' in line:
                typing_line = i
                print(f"   Línea {i}: {line.strip()}")
                
                # Verificar qué está importado
                imports = line.split('import')[1].strip()
                required = ['Dict', 'List', 'Optional', 'Tuple', 'Any']
                missing = [r for r in required if r not in imports]
                
                if missing:
                    print(f"   ❌ Faltan: {', '.join(missing)}")
                else:
                    print(f"   ✅ Todos los imports necesarios presentes")
        
        if not typing_line:
            print("   ❌ NO se encontró 'from typing import'")
        
        print()
        print("2️⃣ Intentando compilar el código...")
        
        try:
            compile(code, filepath, 'exec')
            print("   ✅ Compilación exitosa (sintaxis OK)")
        except SyntaxError as e:
            print(f"   ❌ Error de sintaxis en línea {e.lineno}:")
            print(f"      {e.msg}")
            if e.text:
                print(f"      Código: {e.text.strip()}")
            return False
        
        print()
        print("3️⃣ Intentando ejecutar el código...")
        
        # Crear un namespace limpio
        namespace = {}
        
        try:
            exec(code, namespace)
            print("   ✅ Ejecución exitosa!")
            
            # Verificar qué clases se crearon
            classes = [k for k in namespace.keys() if k[0].isupper() and not k.startswith('_')]
            print(f"\n   Clases creadas ({len(classes)}):")
            for cls in classes:
                print(f"      ✓ {cls}")
            
            # Verificar clases esperadas
            expected = ['MLModelConfig', 'MLResult', 'FeatureEngineer', 'MLEngine', 'MarketRegimeDetector']
            missing = [e for e in expected if e not in classes]
            
            if missing:
                print(f"\n   ❌ Clases faltantes: {', '.join(missing)}")
                return False
            else:
                print(f"\n   ✅ Todas las clases esperadas están presentes")
                return True
            
        except NameError as e:
            print(f"   ❌ Error de nombre: {e}")
            
            # Extraer información del error
            error_msg = str(e)
            if "is not defined" in error_msg:
                undefined = error_msg.split("'")[1]
                print(f"\n   💡 '{undefined}' no está definido")
                print(f"      Probablemente falta importarlo")
                
                # Buscar dónde se usa
                print(f"\n   📍 Buscando dónde se usa '{undefined}'...")
                for i, line in enumerate(lines, 1):
                    if undefined in line and not line.strip().startswith('#'):
                        print(f"      Línea {i}: {line.strip()}")
            
            return False
            
        except Exception as e:
            print(f"   ❌ Error inesperado: {e}")
            import traceback
            print("\n   📋 Traceback completo:")
            traceback.print_exc()
            return False
    
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {filepath}")
        return False
    except Exception as e:
        print(f"❌ Error leyendo archivo: {e}")
        return False

def show_import_section():
    """Mostrar la sección de imports del archivo"""
    print("\n" + "=" * 60)
    print("📄 SECCIÓN DE IMPORTS")
    print("=" * 60)
    print()
    
    try:
        with open('ml/ml_engine.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print("Primeras 20 líneas del archivo:")
        print()
        for i, line in enumerate(lines[:20], 1):
            print(f"{i:3d} | {line.rstrip()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    success = find_all_errors()
    
    show_import_section()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TODO OK - No hay errores detectables")
        print("=" * 60)
        print()
        print("El módulo debería importarse correctamente.")
        print("Si aún hay error, puede ser un problema de cache.")
        print()
        print("Intenta:")
        print("  1. Cerrar todas las ventanas de Python")
        print("  2. Eliminar cache: del ml\\__pycache__ /s /q")
        print("  3. Ejecutar de nuevo: python -m pytest tests/test_suite.py -v")
    else:
        print("❌ ERRORES ENCONTRADOS")
        print("=" * 60)
        print()
        print("Revisa los mensajes arriba para ver qué falta.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())