# main.py
#!/usr/bin/env python3
"""
Plataforma de Trading Algorítmico Avanzada

Sistema completo de trading algorítmico con ML, optimización inteligente
y ejecución en vivo para MetaTrader 5.
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

from core.platform import TradingPlatform, get_platform
from core.monitoring import AdvancedLogger, HealthChecker
from config.deployment import DeploymentManager, DeploymentConfig, BackupManager

def setup_argparse():
    """Configurar parser de argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Plataforma de Trading Algorítmico Avanzada',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Ejemplos de uso:
  {sys.argv[0]} --environment development    # Modo desarrollo
  {sys.argv[0]} --environment production     # Modo producción
  {sys.argv[0]} --deploy --environment production  # Desplegar en producción
  {sys.argv[0]} --backup                     # Crear backup
  {sys.argv[0]} --health-check               # Verificar salud del sistema
        '''
    )
    
    parser.add_argument('--environment', '-e', 
                       choices=['development', 'staging', 'production'],
                       default='development',
                       help='Entorno de ejecución')
    
    parser.add_argument('--log-level', '-l',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Nivel de logging')
    
    parser.add_argument('--deploy', action='store_true',
                       help='Desplegar plataforma')
    
    parser.add_argument('--backup', action='store_true',
                       help='Crear backup de datos y configuraciones')
    
    parser.add_argument('--health-check', action='store_true',
                       help='Ejecutar verificación de salud')
    
    parser.add_argument('--gui', action='store_true',
                       help='Ejecutar interfaz gráfica (por defecto en desarrollo)')
    
    parser.add_argument('--headless', action='store_true',
                       help='Ejecutar sin interfaz gráfica')
    
    return parser.parse_args()

def main():
    """Función principal de la plataforma"""
    args = setup_argparse()
    
    # Configurar logging inicial
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/startup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = AdvancedLogger('Main', MonitoringConfig(log_level=args.log_level))
    logger.logger.info(f"Iniciando plataforma en modo: {args.environment}")
    
    try:
        # Ejecutar acciones según argumentos
        if args.deploy:
            deploy_platform(args.environment, logger)
            return 0
        
        if args.backup:
            create_backup(logger)
            return 0
        
        if args.health_check:
            run_health_check(logger)
            return 0
        
        # Ejecutar plataforma principal
        return run_platform(args, logger)
        
    except KeyboardInterrupt:
        logger.logger.info("Plataforma detenida por el usuario")
        return 0
    except Exception as e:
        logger.logger.error(f"Error crítico: {e}", exc_info=True)
        return 1

def deploy_platform(environment: str, logger: AdvancedLogger):
    """Desplegar plataforma"""
    logger.logger.info(f"Desplegando plataforma en entorno: {environment}")
    
    config = DeploymentConfig(environment=environment)
    deploy_manager = DeploymentManager(config)
    deploy_manager.deploy_platform()
    
    logger.logger.info("Deployment completado exitosamente")

def create_backup(logger: AdvancedLogger):
    """Crear backup del sistema"""
    logger.logger.info("Creando backup del sistema...")
    
    backup_manager = BackupManager()
    backup_manager.create_backup()
    
    logger.logger.info("Backup completado exitosamente")

def run_health_check(logger: AdvancedLogger):
    """Ejecutar verificación de salud"""
    logger.logger.info("Ejecutando verificación de salud...")
    
    platform = TradingPlatform()
    try:
        if platform.initialize():
            health_checker = HealthChecker(platform)
            health_status = health_checker.check_platform_health()
            
            print("\n" + "="*50)
            print("REPORTE DE SALUD DE LA PLATAFORMA")
            print("="*50)
            
            for check_name, check_result in health_status.items():
                status = "✅ SALUDABLE" if check_result.get('healthy') else "❌ PROBLEMA"
                print(f"{check_name.upper():<20} {status}")
                if not check_result.get('healthy'):
                    print(f"   Mensaje: {check_result.get('message', 'N/A')}")
            
            print("="*50)
            
            overall_health = health_status.get('overall_health', {})
            if overall_health.get('healthy'):
                print("✅ PLATAFORMA EN ESTADO SALUDABLE")
                return 0
            else:
                print("❌ PLATAFORMA CON PROBLEMAS")
                failed_checks = overall_health.get('failed_checks', [])
                if failed_checks:
                    print(f"   Checks fallidos: {', '.join(failed_checks)}")
                return 1
        else:
            print("❌ No se pudo inicializar la plataforma para health check")
            return 1
            
    finally:
        platform.shutdown()

def run_platform(args, logger: AdvancedLogger):
    """Ejecutar plataforma principal"""
    platform = None
    
    try:
        # Inicializar plataforma
        platform = get_platform(f"config/config_{args.environment}.yaml")
        
        if not platform.initialize():
            logger.logger.error("No se pudo inicializar la plataforma")
            return 1
        
        logger.logger.info("✅ Plataforma inicializada correctamente")
        
        # Iniciar monitoreo
        logger.start_monitoring()
        
        # Verificación de salud inicial
        health_checker = HealthChecker(platform)
        health_status = health_checker.check_platform_health()
        
        if not health_status.get('overall_health', {}).get('healthy'):
            logger.logger.warning("Problemas detectados en health check inicial")
        
        # Ejecutar interfaz gráfica o modo headless
        if args.headless or args.environment == 'production':
            logger.logger.info("Ejecutando en modo headless...")
            return run_headless_mode(platform, logger)
        else:
            logger.logger.info("Ejecutando interfaz gráfica...")
            from ui.main_window import run_gui
            return run_gui()
    
    except Exception as e:
        logger.logger.error(f"Error ejecutando plataforma: {e}", exc_info=True)
        return 1
    finally:
        if platform and platform.initialized:
            logger.stop_monitoring_system()
            platform.shutdown()

def run_headless_mode(platform: TradingPlatform, logger: AdvancedLogger):
    """Ejecutar plataforma en modo headless (sin GUI)"""
    try:
        logger.logger.info("Iniciando modo headless...")
        
        # Aquí se implementaría la lógica para ejecución en vivo headless
        # Por ejemplo: cargar estrategias, iniciar ejecución en vivo, etc.
        
        # Por ahora, mantener la plataforma corriendo
        import time
        while True:
            time.sleep(60)
            # Verificar salud periódicamente
            health_checker = HealthChecker(platform)
            health_status = health_checker.get_health_status()
            
            if not health_status.get('overall_health', {}).get('healthy'):
                logger.logger.warning("Problemas de salud detectados en modo headless")
    
    except KeyboardInterrupt:
        logger.logger.info("Modo headless detenido por el usuario")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)