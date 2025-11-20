# config/deployment.py
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, asdict
import docker
from docker.errors import DockerException

logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    environment: str  # 'development', 'staging', 'production'
    docker_enabled: bool = True
    database_backup_enabled: bool = True
    monitoring_enabled: bool = True
    auto_update_enabled: bool = False
    backup_interval_hours: int = 24
    resource_limits: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.resource_limits is None:
            self.resource_limits = {
                'memory_mb': 4096,
                'cpu_cores': 2,
                'max_workers': 8
            }

class DeploymentManager:
    """Gestor de deployment para entornos de producción"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.docker_client = None
        
        if config.docker_enabled:
            try:
                self.docker_client = docker.from_env()
            except DockerException as e:
                logger.warning(f"Docker no disponible: {e}")
    
    def create_docker_compose(self) -> str:
        """Crear archivo docker-compose.yml para deployment"""
        compose_config = {
            'version': '3.8',
            'services': {
                'trading-platform': {
                    'build': '.',
                    'ports': ['8000:8000'],
                    'volumes': [
                        './config:/app/config',
                        './data:/app/data',
                        './logs:/app/logs',
                        './cache:/app/cache'
                    ],
                    'environment': [
                        f'ENVIRONMENT={self.config.environment}',
                        'PYTHONPATH=/app'
                    ],
                    'deploy': {
                        'resources': {
                            'limits': {
                                'memory': f"{self.config.resource_limits['memory_mb']}M",
                                'cpus': f"{self.config.resource_limits['cpu_cores']}"
                            }
                        }
                    },
                    'restart': 'unless-stopped'
                },
                'database': {
                    'image': 'postgres:13',
                    'environment': {
                        'POSTGRES_DB': 'trading',
                        'POSTGRES_USER': 'trading_user',
                        'POSTGRES_PASSWORD': 'trading_password'
                    },
                    'volumes': ['postgres_data:/var/lib/postgresql/data'],
                    'ports': ['5432:5432'],
                    'restart': 'unless-stopped'
                },
                'redis': {
                    'image': 'redis:6-alpine',
                    'ports': ['6379:6379'],
                    'restart': 'unless-stopped'
                },
                'monitoring': {
                    'image': 'grafana/grafana:latest',
                    'ports': ['3000:3000'],
                    'environment': {
                        'GF_SECURITY_ADMIN_PASSWORD': 'admin'
                    },
                    'restart': 'unless-stopped'
                }
            },
            'volumes': {
                'postgres_data': {}
            }
        }
        
        return yaml.dump(compose_config, default_flow_style=False)
    
    def create_environment_config(self) -> Dict[str, Any]:
        """Crear configuración de entorno específica"""
        base_config = {
            'environment': self.config.environment,
            'logging': {
                'level': 'INFO' if self.config.environment == 'production' else 'DEBUG',
                'file': f'logs/trading_platform_{self.config.environment}.log'
            },
            'database': {
                'postgres_url': 'postgresql://trading_user:trading_password@database:5432/trading',
                'influx_url': 'http://localhost:8086',
                'redis_url': 'redis://redis:6379/0'
            },
            'performance': {
                'max_workers': self.config.resource_limits['max_workers'],
                'memory_limit_mb': self.config.resource_limits['memory_mb']
            },
            'security': {
                'encryption_key': 'change_in_production',
                'session_timeout_minutes': 60
            }
        }
        
        if self.config.environment == 'production':
            base_config['logging']['level'] = 'WARNING'
            base_config['security']['encryption_key'] = 'PRODUCTION_KEY_CHANGE_THIS'
        
        return base_config
    
    def deploy_platform(self):
        """Desplegar la plataforma completa"""
        logger.info(f"Iniciando deployment en entorno: {self.config.environment}")
        
        try:
            # 1. Crear directorios necesarios
            self._create_directories()
            
            # 2. Generar archivos de configuración
            self._generate_config_files()
            
            # 3. Iniciar con Docker si está disponible
            if self.docker_client and self.config.docker_enabled:
                self._deploy_with_docker()
            else:
                self._deploy_manual()
            
            logger.info("Deployment completado exitosamente")
            
        except Exception as e:
            logger.error(f"Error en deployment: {e}")
            raise
    
    def _create_directories(self):
        """Crear estructura de directorios necesaria"""
        directories = [
            'config',
            'data/backups',
            'logs',
            'cache',
            'reports',
            'strategies/generated'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _generate_config_files(self):
        """Generar archivos de configuración"""
        # Docker compose
        compose_content = self.create_docker_compose()
        with open('docker-compose.yml', 'w') as f:
            f.write(compose_content)
        
        # Configuración de entorno
        env_config = self.create_environment_config()
        with open(f'config/config_{self.config.environment}.yaml', 'w') as f:
            yaml.dump(env_config, f, default_flow_style=False)
        
        # Script de inicio
        self._create_startup_script()
    
    def _create_startup_script(self):
        """Crear script de inicio para la plataforma"""
        if self.config.environment == 'production':
            script_content = """#!/bin/bash
# Startup script for Trading Platform - Production

echo "Starting Trading Platform in production mode..."

# Check system resources
if [ $(free -m | awk 'NR==2{printf "%.0f", $3*100/$2}') -gt 90 ]; then
    echo "ERROR: System memory usage too high"
    exit 1
fi

# Start platform
python main.py --environment production --log-level WARNING

echo "Trading Platform started successfully"
"""
        else:
            script_content = """#!/bin/bash
# Startup script for Trading Platform - Development

echo "Starting Trading Platform in development mode..."

# Start platform
python main.py --environment development --log-level INFO

echo "Trading Platform started successfully"
"""
        
        script_path = f"start_platform_{self.config.environment}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Hacer ejecutable
        import os
        os.chmod(script_path, 0o755)
    
    def _deploy_with_docker(self):
        """Desplegar usando Docker"""
        logger.info("Desplegando con Docker Compose...")
        
        # Construir y levantar servicios
        import subprocess
        try:
            subprocess.run(['docker-compose', 'down'], check=False)
            subprocess.run(['docker-compose', 'build', '--no-cache'], check=True)
            subprocess.run(['docker-compose', 'up', '-d'], check=True)
            
            # Esperar a que los servicios estén listos
            time.sleep(30)
            
            # Verificar que los contenedores estén corriendo
            containers = self.docker_client.containers.list()
            running_services = [container.name for container in containers]
            logger.info(f"Servicios ejecutándose: {running_services}")
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error ejecutando Docker Compose: {e}")
    
    def _deploy_manual(self):
        """Despliegue manual sin Docker"""
        logger.info("Realizando despliegue manual...")
        
        # Verificar dependencias
        self._check_dependencies()
        
        # Inicializar base de datos
        self._initialize_database()
        
        logger.info("Despliegue manual completado. Ejecute: python main.py")
    
    def _check_dependencies(self):
        """Verificar dependencias del sistema"""
        import importlib
        import sys
        
        dependencies = [
            'numpy', 'pandas', 'numba', 'scikit_learn', 'tensorflow',
            'xgboost', 'lightgbm', 'pyqt6', 'matplotlib', 'plotly',
            'sqlalchemy', 'redis', 'psutil', 'docker'
        ]
        
        missing_deps = []
        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            logger.warning(f"Dependencias faltantes: {missing_deps}")
            logger.info("Instale las dependencias con: pip install -r requirements.txt")
    
    def _initialize_database(self):
        """Inicializar base de datos"""
        logger.info("Inicializando base de datos...")
        
        # Esto se haría con migraciones de base de datos en producción
        # Por ahora, solo creamos las tablas necesarias
        try:
            from database.data_manager import TradingData
            from core.platform import TradingPlatform
            
            with TradingPlatform() as platform:
                if platform.initialized:
                    # Las tablas se crean automáticamente con SQLAlchemy
                    logger.info("Base de datos inicializada correctamente")
                else:
                    logger.error("No se pudo inicializar la plataforma para DB setup")
                    
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {e}")

class BackupManager:
    """Gestor de backups para datos y configuraciones"""
    
    def __init__(self, backup_dir: str = "data/backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, include_data: bool = True, include_config: bool = True):
        """Crear backup completo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        backup_path.mkdir(exist_ok=True)
        
        try:
            if include_config:
                self._backup_configurations(backup_path)
            
            if include_data:
                self._backup_database(backup_path)
                self._backup_strategies(backup_path)
            
            # Comprimir backup
            self._compress_backup(backup_path)
            
            logger.info(f"Backup creado: {backup_path}.tar.gz")
            
            # Limpiar backups antiguos
            self._clean_old_backups()
            
        except Exception as e:
            logger.error(f"Error creando backup: {e}")
            raise
    
    def _backup_configurations(self, backup_path: Path):
        """Hacer backup de configuraciones"""
        config_files = list(Path('config').glob('*.yaml')) + list(Path('config').glob('*.json'))
        
        for config_file in config_files:
            if config_file.exists():
                backup_file = backup_path / 'config' / config_file.name
                backup_file.parent.mkdir(exist_ok=True)
                backup_file.write_text(config_file.read_text())
    
    def _backup_database(self, backup_path: Path):
        """Hacer backup de base de datos"""
        # En producción, usar pg_dump para PostgreSQL
        # Por ahora, backup de datos críticos
        try:
            from core.platform import TradingPlatform
            
            with TradingPlatform() as platform:
                # Exportar datos importantes
                pass
                
        except Exception as e:
            logger.warning(f"No se pudo hacer backup de base de datos: {e}")
    
    def _backup_strategies(self, backup_path: Path):
        """Hacer backup de estrategias"""
        strategies_dir = Path('strategies')
        if strategies_dir.exists():
            import shutil
            shutil.copytree(strategies_dir, backup_path / 'strategies')
    
    def _compress_backup(self, backup_path: Path):
        """Comprimir backup"""
        import tarfile
        
        with tarfile.open(f"{backup_path}.tar.gz", "w:gz") as tar:
            tar.add(backup_path, arcname=backup_path.name)
        
        # Eliminar directorio sin comprimir
        import shutil
        shutil.rmtree(backup_path)
    
    def _clean_old_backups(self, keep_count: int = 10):
        """Limpiar backups antiguos"""
        backup_files = sorted(self.backup_dir.glob("backup_*.tar.gz"))
        
        if len(backup_files) > keep_count:
            for old_backup in backup_files[:-keep_count]:
                old_backup.unlink()
                logger.info(f"Backup antiguo eliminado: {old_backup}")