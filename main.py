#!/usr/bin/env python3
# main.py
"""
Plataforma de Trading Algorítmico Avanzada

Sistema completo de trading algorítmico con ML, optimización inteligente
y ejecución en vivo para MetaTrader 5.
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

# Ensure proper imports
sys.path.insert(0, str(Path(__file__).parent))

from core.platform import TradingPlatform, get_platform, reset_platform
from core.monitoring import AdvancedLogger, HealthChecker, MonitoringConfig
from config.deployment import DeploymentManager, DeploymentConfig, BackupManager


def setup_argparse():
    """Configure command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Plataforma de Trading Algorítmico Avanzada',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Examples:
  {sys.argv[0]} --environment development    # Development mode
  {sys.argv[0]} --environment production     # Production mode
  {sys.argv[0]} --deploy --environment production  # Deploy to production
  {sys.argv[0]} --backup                     # Create backup
  {sys.argv[0]} --health-check               # Check system health
  {sys.argv[0]} --lite                       # Force lite mode (SQLite only)
        '''
    )
    
    parser.add_argument('--environment', '-e', 
                       choices=['development', 'staging', 'production'],
                       default='development',
                       help='Execution environment')
    
    parser.add_argument('--log-level', '-l',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')
    
    parser.add_argument('--deploy', action='store_true',
                       help='Deploy platform')
    
    parser.add_argument('--backup', action='store_true',
                       help='Create backup of data and configurations')
    
    parser.add_argument('--health-check', action='store_true',
                       help='Run health check')
    
    parser.add_argument('--gui', action='store_true',
                       help='Run graphical interface')
    
    parser.add_argument('--headless', action='store_true',
                       help='Run without graphical interface')
    
    parser.add_argument('--lite', action='store_true',
                       help='Force lite mode (SQLite, no Redis/InfluxDB)')
    
    parser.add_argument('--test', action='store_true',
                       help='Run test suite')
    
    return parser.parse_args()


def setup_logging(log_level: str) -> AdvancedLogger:
    """Setup logging system."""
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Configure basic logging first
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/startup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create advanced logger
    config = MonitoringConfig(log_level=log_level)
    return AdvancedLogger('Main', config)


def main():
    """Main platform function."""
    args = setup_argparse()
    
    # Setup logging
    adv_logger = setup_logging(args.log_level)
    adv_logger.logger.info(f"Starting platform in {args.environment} mode")
    
    try:
        # Execute actions based on arguments
        if args.deploy:
            return deploy_platform(args.environment, adv_logger)
        
        if args.backup:
            return create_backup(adv_logger)
        
        if args.health_check:
            return run_health_check(adv_logger, args.lite)
        
        if args.test:
            return run_tests(adv_logger)
        
        # Run main platform
        return run_platform(args, adv_logger)
        
    except KeyboardInterrupt:
        adv_logger.logger.info("Platform stopped by user")
        return 0
    except Exception as e:
        adv_logger.logger.error(f"Critical error: {e}", exc_info=True)
        return 1


def deploy_platform(environment: str, logger: AdvancedLogger) -> int:
    """Deploy platform."""
    logger.logger.info(f"Deploying platform in environment: {environment}")
    
    try:
        config = DeploymentConfig(environment=environment)
        deploy_manager = DeploymentManager(config)
        deploy_manager.deploy_platform()
        
        logger.logger.info("Deployment completed successfully")
        return 0
    except Exception as e:
        logger.logger.error(f"Deployment failed: {e}")
        return 1


def create_backup(logger: AdvancedLogger) -> int:
    """Create system backup."""
    logger.logger.info("Creating system backup...")
    
    try:
        backup_manager = BackupManager()
        archive_path = backup_manager.create_backup()
        
        logger.logger.info(f"Backup completed: {archive_path}")
        return 0
    except Exception as e:
        logger.logger.error(f"Backup failed: {e}")
        return 1


def run_health_check(logger: AdvancedLogger, lite_mode: bool = False) -> int:
    """Run health check."""
    logger.logger.info("Running health check...")
    
    platform = TradingPlatform(lite_mode=lite_mode)
    try:
        if platform.initialize():
            health_checker = HealthChecker(platform)
            health_status = health_checker.check_platform_health()
            
            print("\n" + "="*50)
            print("PLATFORM HEALTH REPORT")
            print("="*50)
            
            for check_name, check_result in health_status.items():
                if check_name == 'overall_health':
                    continue
                status = "✅ HEALTHY" if check_result.get('healthy') else "❌ ISSUE"
                print(f"{check_name.upper():<25} {status}")
                if not check_result.get('healthy'):
                    print(f"   Message: {check_result.get('message', 'N/A')}")
            
            print("="*50)
            
            overall_health = health_status.get('overall_health', {})
            if overall_health.get('healthy'):
                print("✅ PLATFORM IS HEALTHY")
                return 0
            else:
                print("❌ PLATFORM HAS ISSUES")
                failed_checks = overall_health.get('failed_checks', [])
                if failed_checks:
                    print(f"   Failed checks: {', '.join(failed_checks)}")
                return 1
        else:
            print("❌ Could not initialize platform for health check")
            return 1
            
    finally:
        platform.shutdown()


def run_tests(logger: AdvancedLogger) -> int:
    """Run test suite."""
    logger.logger.info("Running test suite...")
    
    try:
        from tests.test_suite import run_all_tests
        success = run_all_tests()
        return 0 if success else 1
    except ImportError as e:
        logger.logger.error(f"Could not import test suite: {e}")
        return 1
    except Exception as e:
        logger.logger.error(f"Test suite failed: {e}")
        return 1


def run_platform(args, logger: AdvancedLogger) -> int:
    """Run main platform."""
    platform = None
    
    try:
        # Initialize platform
        config_path = f"config/config_{args.environment}.yaml"
        if not Path(config_path).exists():
            config_path = "config/platform_config.yaml"
        
        platform = get_platform(config_path, lite_mode=args.lite)
        
        if not platform.initialize():
            logger.logger.warning("Platform initialized with warnings")
        
        logger.logger.info("✅ Platform initialized successfully")
        
        # Start monitoring
        logger.start_monitoring()
        
        # Initial health check
        health_checker = HealthChecker(platform)
        health_status = health_checker.check_platform_health()
        
        if not health_status.get('overall_health', {}).get('healthy'):
            logger.logger.warning("Issues detected in initial health check")
        
        # Run GUI or headless mode
        if args.headless or args.environment == 'production':
            logger.logger.info("Running in headless mode...")
            return run_headless_mode(platform, logger)
        else:
            logger.logger.info("Running graphical interface...")
            return run_gui_mode(platform, logger)
    
    except Exception as e:
        logger.logger.error(f"Platform execution error: {e}", exc_info=True)
        return 1
    finally:
        if platform and platform.initialized:
            logger.stop_monitoring_system()
            platform.shutdown()


def run_gui_mode(platform: TradingPlatform, logger: AdvancedLogger) -> int:
    """Run platform with GUI."""
    try:
        from ui.main_window import run_gui
        return run_gui()
    except ImportError as e:
        logger.logger.warning(f"GUI not available: {e}")
        logger.logger.info("Falling back to headless mode...")
        return run_headless_mode(platform, logger)
    except Exception as e:
        logger.logger.error(f"GUI error: {e}")
        return 1


def run_headless_mode(platform: TradingPlatform, logger: AdvancedLogger) -> int:
    """Run platform in headless mode (no GUI)."""
    try:
        logger.logger.info("Starting headless mode...")
        logger.logger.info("Press Ctrl+C to stop")
        
        health_checker = HealthChecker(platform)
        check_interval = 60  # Check health every 60 seconds
        
        while True:
            time.sleep(check_interval)
            
            # Periodic health check
            health_status = health_checker.get_health_status()
            
            if not health_status.get('overall_health', {}).get('healthy'):
                logger.logger.warning("Health issues detected in headless mode")
                failed_checks = health_status.get('overall_health', {}).get('failed_checks', [])
                for check in failed_checks:
                    logger.logger.warning(f"  - {check}: {health_status.get(check, {}).get('message', 'Unknown')}")
    
    except KeyboardInterrupt:
        logger.logger.info("Headless mode stopped by user")
        return 0


def quick_start():
    """Quick start function for basic usage."""
    print("🚀 Starting Trading Platform...")
    
    # Create platform in lite mode for easy setup
    platform = TradingPlatform(lite_mode=True)
    
    if platform.initialize():
        print("✅ Platform initialized successfully!")
        print(f"   Mode: {'Lite (SQLite)' if platform.lite_mode else 'Full'}")
        print(f"   MT5: {'Connected' if platform.mt5_connector and platform.mt5_connector.connected else 'Offline'}")
        
        # Get account info
        account = platform.get_account_summary()
        print(f"   Account: {account.get('login', 'Demo')}")
        print(f"   Balance: ${account.get('balance', 0):,.2f}")
        
        return platform
    else:
        print("❌ Platform initialization failed")
        return None


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
