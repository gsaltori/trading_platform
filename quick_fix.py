#!/usr/bin/env python3
"""
Fix ultra-simple: Solo arregla el import sin tocar nada más
"""

import os

def quick_fix():
    """Agregar imports faltantes al inicio del archivo"""
    
    filepath = "ml/ml_engine.py"
    
    if not os.path.exists(filepath):
        print(f"❌ Archivo no encontrado: {filepath}")
        return False
    
    print("🔧 Aplicando fix ultra-simple...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar si falta el import de Tuple
    if 'from typing import' in content and 'Tuple' not in content.split('from typing import')[1].split('\n')[0]:
        print("   ⚠️  Falta import de Tuple")
        
        # Encontrar la línea de imports de typing
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('from typing import') and 'Tuple' not in line:
                # Agregar Tuple al import
                lines[i] = line.rstrip() + ', Tuple'
                print(f"   ✓ Agregado Tuple al import en línea {i+1}")
                content = '\n'.join(lines)
                break
    
    # Verificar otros imports comunes que pueden faltar
    required_imports = {
        'import ta': 'ta',
        'from ta import': 'ta',
        'from ta.': 'ta',
    }
    
    # Verificar si ta está importado
    has_ta = any(imp in content for imp in required_imports.keys())
    
    if not has_ta:
        print("   ⚠️  Falta import de ta (indicadores técnicos)")
        print("   💡 Agregando imports de ta...")
        
        # Agregar imports de ta después de los imports de sklearn
        lines = content.split('\n')
        insert_index = None
        
        for i, line in enumerate(lines):
            if 'from sklearn' in line or 'import sklearn' in line:
                insert_index = i + 1
        
        if insert_index:
            ta_imports = [
                'import ta',
                'from ta import add_all_ta_features',
                'from ta.momentum import RSIIndicator, StochasticOscillator',
                'from ta.trend import MACD, EMAIndicator, ADXIndicator',
                'from ta.volatility import BollingerBands, AverageTrueRange',
            ]
            
            # Insertar imports
            for imp in reversed(ta_imports):
                lines.insert(insert_index, imp)
            
            content = '\n'.join(lines)
            print(f"   ✓ Agregados imports de ta")
    
    # Guardar
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fix aplicado")
    
    # Verificar que compila
    try:
        compile(content, filepath, 'exec')
        print("✅ Sintaxis correcta")
        return True
    except SyntaxError as e:
        print(f"❌ Error de sintaxis: {e}")
        return False

def main():
    print("=" * 60)
    print("🚀 FIX ULTRA-SIMPLE")
    print("=" * 60)
    print()
    
    if quick_fix():
        print()
        print("✅ FIX APLICADO")
        print()
        print("Ahora ejecuta:")
        print("  python -m pytest tests/test_suite.py::TestTradingPlatform::test_05_machine_learning_engine -v")
        return 0
    else:
        print()
        print("❌ FIX FALLÓ")
        print()
        print("Ejecuta para más información:")
        print("  python deep_diagnose.py")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())