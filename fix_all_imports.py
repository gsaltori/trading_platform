#!/usr/bin/env python3
# fix_all_imports.py - Script automático para corregir todos los imports

import sys
from pathlib import Path
import re

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_color(message, color=Colors.GREEN):
    print(f"{color}{message}{Colors.ENDC}")

def fix_main_py():
    """Corregir imports en main.py"""
    print_color("🔧 Corrigiendo main.py...", Colors.BLUE)
    
    main_path = Path('main.py')
    
    if not main_path.exists():
        print_color("❌ main.py no encontrado", Colors.RED)
        return False
    
    try:
        content = main_path.read_text(encoding='utf-8')
        original_content = content
        
        # Corrección 1: Agregar MonitoringConfig al import
        patterns_to_fix = [
            (
                r'from core\.monitoring import AdvancedLogger, HealthChecker(?!,\s*MonitoringConfig)',
                'from core.monitoring import AdvancedLogger, HealthChecker, MonitoringConfig'
            ),
        ]
        
        for old_pattern, new_text in patterns_to_fix:
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_text, content)
                print_color(f"  ✅ Agregado MonitoringConfig al import", Colors.GREEN)
        
        # Solo escribir si hubo cambios
        if content != original_content:
            main_path.write_text(content, encoding='utf-8')
            print_color("✅ main.py corregido exitosamente", Colors.GREEN)
        else:
            print_color("✅ main.py ya estaba correcto", Colors.GREEN)
        
        return True
        
    except Exception as e:
        print_color(f"❌ Error corrigiendo main.py: {e}", Colors.RED)
        return False

def create_init_files():
    """Crear archivos __init__.py faltantes en todos los módulos"""
    print_color("\n🔧 Creando archivos __init__.py...", Colors.BLUE)
    
    modules = {
        'config': '',
        'core': '',
        'risk_management': '''# risk_management/__init__.py
"""Risk Management Module"""
from .risk_engine import RiskEngine, RiskParameters

__all__ = ['RiskEngine', 'RiskParameters']
''',
        'strategies': '''# strategies/__init__.py
"""Strategies Module"""
from .strategy_engine import StrategyEngine, StrategyConfig, BaseStrategy

__all__ = ['StrategyEngine', 'StrategyConfig', 'BaseStrategy']
''',
        'backtesting': '''# backtesting/__init__.py
"""Backtesting Module"""
from .backtest_engine import BacktestEngine, BacktestResult

__all__ = ['BacktestEngine', 'BacktestResult']
''',
        'execution': '''# execution/__init__.py
"""Execution Module"""
from .live_execution import LiveExecutionEngine, LiveTradingConfig

__all__ = ['LiveExecutionEngine', 'LiveTradingConfig']
''',
        'ml': '''# ml/__init__.py
"""Machine Learning Module"""
from .ml_engine import MLEngine, MLModelConfig

__all__ = ['MLEngine', 'MLModelConfig']
''',
        'data': '''# data/__init__.py
"""Data Module"""
from .mt5_connector import MT5ConnectionManager

__all__ = ['MT5ConnectionManager']
''',
        'database': '''# database/__init__.py
"""Database Module"""
from .data_manager import DataManager

__all__ = ['DataManager']
''',
        'optimization': '''# optimization/__init__.py
"""Optimization Module"""
from .genetic_optimizer import GeneticOptimizer, OptimizationConfig

__all__ = ['GeneticOptimizer', 'OptimizationConfig']
''',
        'tests': '',
        'ui': '',
    }
    
    created_count = 0
    for module_name, content in modules.items():
        module_path = Path(module_name)
        
        # Crear directorio si no existe
        if not module_path.exists():
            module_path.mkdir(parents=True, exist_ok=True)
            print_color(f"  📁 Creado directorio: {module_name}/", Colors.YELLOW)
        
        init_file = module_path / '__init__.py'
        
        if not init_file.exists() or init_file.stat().st_size == 0:
            init_file.write_text(content if content else '# ' + module_name + '\n', encoding='utf-8')
            print_color(f"  ✅ Creado: {module_name}/__init__.py", Colors.GREEN)
            created_count += 1
        else:
            # Verificar si necesita actualización
            current_content = init_file.read_text(encoding='utf-8')
            if content and content not in current_content and len(content) > 10:
                # Backup del archivo original
                backup_file = init_file.with_suffix('.py.backup')
                backup_file.write_text(current_content, encoding='utf-8')
                
                init_file.write_text(content, encoding='utf-8')
                print_color(f"  🔄 Actualizado: {module_name}/__init__.py (backup creado)", Colors.YELLOW)
                created_count += 1
    
    if created_count > 0:
        print_color(f"✅ {created_count} archivos __init__.py creados/actualizados", Colors.GREEN)
    else:
        print_color("✅ Todos los archivos __init__.py ya existen", Colors.GREEN)
    
    return True

def verify_imports():
    """Verificar que todos los imports funcionen"""
    print_color("\n🧪 Verificando imports...", Colors.BLUE)
    
    tests = [
        ('core.monitoring', 'MonitoringConfig'),
        ('risk_management', 'RiskEngine'),
        ('strategies', 'StrategyEngine'),
        ('backtesting', 'BacktestEngine'),
        ('execution', 'LiveExecutionEngine'),
    ]
    
    all_ok = True
    for module, item in tests:
        try:
            exec(f"from {module} import {item}")
            print_color(f"  ✅ {module}.{item}", Colors.GREEN)
        except ImportError as e:
            print_color(f"  ❌ {module}.{item} - {str(e)}", Colors.RED)
            all_ok = False
        except Exception as e:
            print_color(f"  ⚠️  {module}.{item} - {str(e)}", Colors.YELLOW)
    
    return all_ok

def create_missing_files():
    """Crear archivos faltantes básicos"""
    print_color("\n🔧 Verificando archivos faltantes...", Colors.BLUE)
    
    # Verificar si existen los archivos principales
    critical_files = [
        'core/platform.py',
        'core/monitoring.py',
        'risk_management/risk_engine.py',
        'config/settings.py',
        'config/deployment.py',
    ]
    
    missing = []
    for file_path in critical_files:
        if not Path(file_path).exists():
            missing.append(file_path)
            print_color(f"  ⚠️  Falta: {file_path}", Colors.YELLOW)
    
    if missing:
        print_color(f"\n⚠️  {len(missing)} archivos críticos faltan:", Colors.YELLOW)
        for f in missing:
            print(f"     - {f}")
        print_color("\n💡 Descarga los archivos corregidos de /mnt/user-data/outputs/", Colors.BLUE)
        return False
    else:
        print_color("✅ Todos los archivos críticos existen", Colors.GREEN)
        return True

def main():
    print_color("\n" + "="*60, Colors.BOLD)
    print_color("   CORRECCIÓN AUTOMÁTICA DE IMPORTS", Colors.BOLD)
    print_color("="*60 + "\n", Colors.BOLD)
    
    # Verificar que estamos en el directorio correcto
    if not Path('main.py').exists():
        print_color("❌ Error: main.py no encontrado", Colors.RED)
        print_color("   Asegúrate de ejecutar este script desde el directorio raíz del proyecto", Colors.YELLOW)
        return False
    
    success = True
    
    # Paso 1: Crear directorios y __init__.py
    if not create_init_files():
        success = False
    
    # Paso 2: Corregir main.py
    if not fix_main_py():
        success = False
    
    # Paso 3: Verificar archivos críticos
    if not create_missing_files():
        success = False
    
    # Paso 4: Verificar imports
    print_color("\n🔍 Verificación final...", Colors.BLUE)
    if verify_imports():
        print_color("✅ Todos los imports verificados exitosamente", Colors.GREEN)
    else:
        print_color("⚠️  Algunos imports fallaron (ver arriba)", Colors.YELLOW)
        success = False
    
    # Resumen
    print_color("\n" + "="*60, Colors.BOLD)
    if success:
        print_color("✅ ¡CORRECCIONES APLICADAS EXITOSAMENTE!", Colors.GREEN)
        print_color("\n📋 Próximos pasos:", Colors.BLUE)
        print("   1. python main.py --health-check")
        print("   2. python -m pytest tests/ -v")
        print("   3. python main.py --environment development")
    else:
        print_color("⚠️  CORRECCIONES PARCIALES", Colors.YELLOW)
        print_color("\n📋 Acción requerida:", Colors.BLUE)
        print("   1. Revisa los errores arriba")
        print("   2. Descarga archivos corregidos de:")
        print("      /mnt/user-data/outputs/")
        print("   3. Copia los archivos faltantes a tu proyecto")
    print_color("="*60 + "\n", Colors.BOLD)
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_color("\n\n❌ Cancelado por el usuario", Colors.RED)
        sys.exit(1)
    except Exception as e:
        print_color(f"\n\n❌ Error crítico: {e}", Colors.RED)
        import traceback
        traceback.print_exc()
        sys.exit(1)