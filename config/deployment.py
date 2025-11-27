# config/deployment.py
"""
Deployment management for the trading platform.

Handles Docker deployment, environment configuration, and backups.
"""

import yaml
import json
import time
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""
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
    """Manages deployment for production environments."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.docker_client = None
        
        if config.docker_enabled:
            try:
                import docker
                self.docker_client = docker.from_env()
            except ImportError:
                logger.warning("Docker package not installed")
            except Exception as e:
                logger.warning(f"Docker not available: {e}")
    
    def create_docker_compose(self) -> str:
        """Create docker-compose.yml for deployment."""
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
                }
            }
        }
        
        # Add database services for non-lite deployments
        if self.config.environment in ['staging', 'production']:
            compose_config['services']['database'] = {
                'image': 'postgres:13',
                'environment': {
                    'POSTGRES_DB': 'trading',
                    'POSTGRES_USER': 'trading_user',
                    'POSTGRES_PASSWORD': 'trading_password'
                },
                'volumes': ['postgres_data:/var/lib/postgresql/data'],
                'ports': ['5432:5432'],
                'restart': 'unless-stopped'
            }
            
            compose_config['services']['redis'] = {
                'image': 'redis:6-alpine',
                'ports': ['6379:6379'],
                'restart': 'unless-stopped'
            }
            
            compose_config['volumes'] = {'postgres_data': {}}
        
        # Add monitoring for production
        if self.config.environment == 'production' and self.config.monitoring_enabled:
            compose_config['services']['monitoring'] = {
                'image': 'grafana/grafana:latest',
                'ports': ['3000:3000'],
                'environment': {
                    'GF_SECURITY_ADMIN_PASSWORD': 'admin'
                },
                'restart': 'unless-stopped'
            }
        
        return yaml.dump(compose_config, default_flow_style=False)
    
    def create_environment_config(self) -> Dict[str, Any]:
        """Create environment-specific configuration."""
        base_config = {
            'environment': self.config.environment,
            'logging': {
                'level': 'INFO' if self.config.environment == 'production' else 'DEBUG',
                'file': f'logs/trading_platform_{self.config.environment}.log'
            },
            'database': {
                'postgres_url': 'postgresql://trading_user:trading_password@database:5432/trading' 
                               if self.config.environment != 'development' 
                               else 'sqlite:///data/sqlite/trading.db',
                'influx_url': 'http://localhost:8086',
                'redis_url': 'redis://redis:6379/0' if self.config.environment != 'development' else ''
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
        """Deploy the complete platform."""
        logger.info(f"Starting deployment in environment: {self.config.environment}")
        
        try:
            # 1. Create necessary directories
            self._create_directories()
            
            # 2. Generate configuration files
            self._generate_config_files()
            
            # 3. Deploy with Docker if available
            if self.docker_client and self.config.docker_enabled:
                self._deploy_with_docker()
            else:
                self._deploy_manual()
            
            logger.info("Deployment completed successfully")
            
        except Exception as e:
            logger.error(f"Deployment error: {e}")
            raise
    
    def _create_directories(self):
        """Create necessary directory structure."""
        directories = [
            'config',
            'data/backups',
            'data/sqlite',
            'logs',
            'cache',
            'reports',
            'strategies/generated'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _generate_config_files(self):
        """Generate configuration files."""
        # Docker compose
        compose_content = self.create_docker_compose()
        with open('docker-compose.yml', 'w') as f:
            f.write(compose_content)
        
        # Environment configuration
        env_config = self.create_environment_config()
        with open(f'config/config_{self.config.environment}.yaml', 'w') as f:
            yaml.dump(env_config, f, default_flow_style=False)
        
        # Startup script
        self._create_startup_script()
        
        # Dockerfile
        self._create_dockerfile()
    
    def _create_startup_script(self):
        """Create startup script for the platform."""
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
        
        script_path = Path(f"start_platform_{self.config.environment}.sh")
        script_path.write_text(script_content)
        
        # Make executable on Unix systems
        try:
            import os
            os.chmod(script_path, 0o755)
        except Exception:
            pass  # Windows doesn't support chmod
    
    def _create_dockerfile(self):
        """Create Dockerfile for containerized deployment."""
        dockerfile_content = """FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/sqlite data/cache reports

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "main.py", "--environment", "production"]
"""
        
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile_content)
    
    def _deploy_with_docker(self):
        """Deploy using Docker Compose."""
        logger.info("Deploying with Docker Compose...")
        
        import subprocess
        try:
            subprocess.run(['docker-compose', 'down'], check=False, capture_output=True)
            subprocess.run(['docker-compose', 'build', '--no-cache'], check=True)
            subprocess.run(['docker-compose', 'up', '-d'], check=True)
            
            # Wait for services to be ready
            time.sleep(30)
            
            # Verify containers are running
            containers = self.docker_client.containers.list()
            running_services = [container.name for container in containers]
            logger.info(f"Running services: {running_services}")
            
        except FileNotFoundError:
            logger.warning("docker-compose not found, falling back to manual deployment")
            self._deploy_manual()
        except subprocess.CalledProcessError as e:
            raise Exception(f"Docker Compose execution error: {e}")
    
    def _deploy_manual(self):
        """Manual deployment without Docker."""
        logger.info("Performing manual deployment...")
        
        # Check dependencies
        self._check_dependencies()
        
        logger.info("Manual deployment completed. Run: python main.py")
    
    def _check_dependencies(self):
        """Check system dependencies."""
        import importlib
        
        dependencies = [
            'numpy', 'pandas', 'scikit_learn',
            'matplotlib', 'plotly', 'sqlalchemy', 'psutil'
        ]
        
        missing_deps = []
        for dep in dependencies:
            try:
                importlib.import_module(dep.replace('-', '_'))
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            logger.warning(f"Missing dependencies: {missing_deps}")
            logger.info("Install with: pip install -r requirements.txt")


class BackupManager:
    """Manages backups for data and configurations."""
    
    def __init__(self, backup_dir: str = "data/backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, include_data: bool = True, include_config: bool = True):
        """Create a complete backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        backup_path.mkdir(exist_ok=True)
        
        try:
            if include_config:
                self._backup_configurations(backup_path)
            
            if include_data:
                self._backup_database(backup_path)
                self._backup_strategies(backup_path)
            
            # Compress backup
            archive_path = self._compress_backup(backup_path)
            
            logger.info(f"Backup created: {archive_path}")
            
            # Clean old backups
            self._clean_old_backups()
            
            return archive_path
            
        except Exception as e:
            logger.error(f"Backup creation error: {e}")
            raise
    
    def _backup_configurations(self, backup_path: Path):
        """Backup configuration files."""
        config_dir = Path('config')
        if not config_dir.exists():
            return
        
        backup_config_dir = backup_path / 'config'
        backup_config_dir.mkdir(exist_ok=True)
        
        for config_file in config_dir.glob('*.yaml'):
            shutil.copy2(config_file, backup_config_dir / config_file.name)
        
        for config_file in config_dir.glob('*.json'):
            shutil.copy2(config_file, backup_config_dir / config_file.name)
    
    def _backup_database(self, backup_path: Path):
        """Backup database files."""
        # Backup SQLite database
        sqlite_db = Path('data/sqlite/trading.db')
        if sqlite_db.exists():
            backup_db_dir = backup_path / 'database'
            backup_db_dir.mkdir(exist_ok=True)
            shutil.copy2(sqlite_db, backup_db_dir / 'trading.db')
    
    def _backup_strategies(self, backup_path: Path):
        """Backup strategy files."""
        strategies_dir = Path('strategies')
        if strategies_dir.exists():
            shutil.copytree(
                strategies_dir, 
                backup_path / 'strategies',
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc')
            )
    
    def _compress_backup(self, backup_path: Path) -> Path:
        """Compress backup directory."""
        archive_path = backup_path.with_suffix('.tar.gz')
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(backup_path, arcname=backup_path.name)
        
        # Remove uncompressed directory
        shutil.rmtree(backup_path)
        
        return archive_path
    
    def _clean_old_backups(self, keep_count: int = 10):
        """Clean old backup files."""
        backup_files = sorted(self.backup_dir.glob("backup_*.tar.gz"))
        
        if len(backup_files) > keep_count:
            for old_backup in backup_files[:-keep_count]:
                old_backup.unlink()
                logger.info(f"Deleted old backup: {old_backup}")
    
    def restore_backup(self, backup_path: str):
        """Restore from a backup archive."""
        backup_file = Path(backup_path)
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # Extract backup
        extract_dir = backup_file.parent / 'restore_temp'
        extract_dir.mkdir(exist_ok=True)
        
        with tarfile.open(backup_file, "r:gz") as tar:
            tar.extractall(extract_dir)
        
        # Restore files
        extracted_backup = list(extract_dir.iterdir())[0]
        
        # Restore config
        config_backup = extracted_backup / 'config'
        if config_backup.exists():
            for config_file in config_backup.glob('*'):
                shutil.copy2(config_file, Path('config') / config_file.name)
        
        # Restore database
        db_backup = extracted_backup / 'database' / 'trading.db'
        if db_backup.exists():
            Path('data/sqlite').mkdir(parents=True, exist_ok=True)
            shutil.copy2(db_backup, Path('data/sqlite/trading.db'))
        
        # Cleanup
        shutil.rmtree(extract_dir)
        
        logger.info(f"Backup restored from: {backup_path}")
    
    def list_backups(self) -> list:
        """List available backups."""
        backups = []
        for backup_file in sorted(self.backup_dir.glob("backup_*.tar.gz"), reverse=True):
            stat = backup_file.stat()
            backups.append({
                'name': backup_file.name,
                'path': str(backup_file),
                'size_mb': stat.st_size / 1024 / 1024,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat()
            })
        return backups
