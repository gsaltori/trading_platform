# Plataforma de Trading Algorítmico Avanzada - VERSIÓN CORREGIDA

Sistema completo de trading algorítmico de nivel empresarial con Machine Learning, optimización inteligente y ejecución en vivo para MetaTrader 5.

> **⚠️ IMPORTANTE**: Esta versión incluye todas las correcciones críticas y mejoras de robustez necesarias para producción.

## 🚀 Características Principales

### 🤖 Machine Learning Avanzado
- **Múltiples algoritmos**: XGBoost, Random Forest, LSTM, Ensemble
- **Feature engineering automático**: 50+ indicadores técnicos
- **Detección de regímenes de mercado**
- **Modelos predictivos** para dirección de precios
- **Training paralelo** para múltiples símbolos

### 🔧 Optimización Inteligente
- **Algoritmos genéticos mejorados** con NSGA-II
- **Optimización bayesiana**
- **Multi-objetivo** (Sharpe vs Drawdown)
- **Paralelización masiva** con multiprocessing
- **Validación cruzada** para evitar overfitting

### 📊 Backtesting de Alta Performance
- **Vectorizado con Numba** (10-50x más rápido)
- **Ejecución realista** (slippage dinámico, spread variable)
- **Métricas avanzadas** (Sharpe, Sortino, Calmar, Omega, Recovery Factor)
- **Walk-forward analysis** automatizado
- **Monte Carlo simulations** para validación

### ⚡ Ejecución en Vivo
- **Conexión nativa MT5** con reconexión automática
- **Gestión automática de órdenes** con confirmación
- **Risk management en tiempo real**
- **Circuit breakers inteligentes** anti-catástrofe
- **Filtros avanzados** (volatilidad, noticias, correlaciones)

### 🎨 Interfaz Moderna
- **Dashboard en tiempo real** con métricas actualizadas
- **Editor visual de estrategias**
- **Gráficos interactivos** con Plotly
- **Modo oscuro/claro** personalizable
- **Alertas visuales y sonoras**

### 🔒 Robustez y Seguridad
- **Thread-safe** en todos los componentes críticos
- **Validación de datos** automática
- **Manejo de errores** comprehensivo
- **Backup automático** con múltiples destinos
- **Encriptación de credenciales**
- **Auditoría completa** de operaciones

## 📋 Requisitos del Sistema

### Mínimos
- Python 3.10+
- 8GB RAM
- 10GB espacio libre en disco
- Windows 10/11 o Linux (Ubuntu 20.04+)

### Recomendados
- Python 3.11
- 16GB RAM
- 50GB SSD
- Procesador multi-core (4+ cores)
- GPU (opcional, para ML avanzado)

## 🛠 Instalación

### Instalación Rápida

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/trading-platform.git
cd trading-platform

# 2. Ejecutar instalación automatizada
python install_complete.py
```

### Instalación Manual

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 2. Actualizar pip
pip install --upgrade pip

# 3. Instalar dependencias core
pip install numpy pandas scikit-learn numba scipy psutil pyyaml joblib tqdm

# 4. Instalar dependencias de trading
pip install MetaTrader5 yfinance ccxt

# 5. Instalar ML (opcional pero recomendado)
pip install tensorflow-cpu xgboost lightgbm

# 6. Instalar GUI
pip install PyQt6 PyQt6-WebEngine qt-material

# 7. Instalar visualización
pip install matplotlib plotly seaborn

# 8. Instalar bases de datos
pip install sqlalchemy redis psycopg2-binary influxdb-client

# 9. Instalar optimización
pip install bayesian-optimization deap

# 10. Instalar indicadores técnicos (alternativas)
pip install ta  # O: pip install pandas-ta
```

### Configuración Inicial

```bash
# 1. Copiar archivo de configuración
cp .env.example .env

# 2. Editar configuración
nano .env  # O tu editor preferido

# 3. Configurar MT5
# Edita config/platform_config.yaml con tus credenciales

# 4. Crear estructura de base de datos (opcional)
python -c "from database.data_manager import DataManager; DataManager.init_db()"
```

## 🚦 Inicio Rápido

### Modo Desarrollo (con GUI)

```bash
python main.py --environment development --gui
```

### Modo Headless (sin GUI)

```bash
python main.py --environment production --headless
```

### Ejecutar Tests

```bash
# Tests completos
python -m pytest tests/ -v

# Tests específicos
python -m pytest tests/test_suite.py::TestTradingPlatform::test_platform_initialization

# Tests con coverage
pytest --cov=. --cov-report=html tests/
```

### Verificar Salud del Sistema

```bash
python main.py --health-check
```

## 📖 Uso Básico

### 1. Crear una Estrategia Simple

```python
from strategies.strategy_engine import StrategyEngine, StrategyConfig

# Crear configuración
config = StrategyConfig(
    name="MA_Crossover_Simple",
    symbols=["EURUSD", "GBPUSD"],
    timeframe="H1",
    parameters={
        'fast_period': 10,
        'slow_period': 20,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    },
    risk_management={
        'atr_multiplier': 2.0,
        'risk_reward_ratio': 1.5
    }
)

# Crear estrategia
engine = StrategyEngine()
strategy = engine.create_strategy('ma_crossover', config)
```

### 2. Ejecutar Backtest

```python
from backtesting.backtest_engine import BacktestEngine
from core.platform import get_platform

# Obtener datos
platform = get_platform()
platform.initialize()
data = platform.get_market_data("EURUSD", "H1", days=365)

# Ejecutar backtest
backtest_engine = BacktestEngine(initial_capital=10000)
result = backtest_engine.run_backtest(
    data=data,
    strategy=strategy,
    symbol="EURUSD",
    commission=0.001,
    slippage_model="dynamic"
)

# Mostrar resultados
print(f"Retorno Total: {result.total_return:.2f}%")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2f}%")
print(f"Win Rate: {result.win_rate:.1f}%")
```

### 3. Optimizar Parámetros

```python
from optimization.genetic_optimizer import GeneticOptimizer, OptimizationConfig

# Configurar optimización
opt_config = OptimizationConfig(
    strategy_name="MA_Crossover_Simple",
    parameter_ranges={
        'fast_period_int': (5, 20),
        'slow_period_int': (20, 50),
        'rsi_period_int': (10, 20)
    },
    objective='sharpe',
    population_size=50,
    generations=30
)

# Ejecutar optimización
optimizer = GeneticOptimizer(backtest_engine)
result = optimizer.optimize_strategy(strategy, data, opt_config)

print(f"Mejores parámetros: {result['best_parameters']}")
print(f"Mejor Sharpe: {result['best_fitness']:.2f}")
```

### 4. Trading en Vivo

```python
from execution.live_execution import LiveExecutionEngine, LiveTradingConfig

# Configurar trading en vivo
live_config = LiveTradingConfig(
    strategy_name="MA_Crossover_Optimized",
    symbols=["EURUSD"],
    timeframe="H1",
    enabled=True,
    max_positions=3,
    risk_per_trade=0.02,
    daily_loss_limit=0.05,
    max_drawdown=0.15
)

# Iniciar motor de ejecución
live_engine = LiveExecutionEngine()
live_engine.add_strategy(live_config)
live_engine.start_trading()

# Monitorear
while True:
    status = live_engine.get_portfolio_status()
    print(f"P&L Diario: {status['daily_pnl']:.2f}")
    print(f"Posiciones Abiertas: {status['open_positions']}")
    time.sleep(60)
```

## 🔧 Configuración Avanzada

### Base de Datos

```yaml
# config/platform_config.yaml
database:
  postgres_url: "postgresql://user:pass@localhost:5432/trading"
  redis_url: "redis://localhost:6379/0"
  influx_url: "http://localhost:8086"
  influx_token: "your-token"
  influx_org: "trading"
```

### Alertas

```yaml
# config/platform_config.yaml
alerts:
  email:
    enabled: true
    smtp_host: "smtp.gmail.com"
    smtp_port: 587
    from: "trading@example.com"
    to: "alerts@example.com"
  
  telegram:
    enabled: true
    bot_token: "your-bot-token"
    chat_id: "your-chat-id"
  
  webhook:
    enabled: true
    url: "https://your-webhook-url.com/alerts"
```

### Risk Management

```yaml
# config/platform_config.yaml
risk:
  max_drawdown: 0.15  # 15%
  max_position_size: 0.10  # 10% del capital
  daily_loss_limit: 0.05  # 5% diario
  correlation_threshold: 0.7  # Máxima correlación entre posiciones
  max_simultaneous_positions: 5
  use_kelly_criterion: false
  use_volatility_sizing: true
```

## 🐳 Deployment con Docker

### Docker Compose

```bash
# Iniciar todos los servicios
docker-compose -f docker-compose.production.yml up -d

# Ver logs
docker-compose -f docker-compose.production.yml logs -f

# Detener
docker-compose -f docker-compose.production.yml down
```

### Script de Deployment

```bash
# Deployment en staging
./deploy.sh staging

# Deployment en producción
./deploy.sh production

# Con backup previo
./deploy.sh production --with-backup
```

## 📊 Monitoring y Observabilidad

### Prometheus Metrics

```bash
# Métricas disponibles en:
http://localhost:9090/metrics

# Principales métricas:
# - trading_trades_total
# - trading_account_balance
# - trading_drawdown_percent
# - trading_open_positions
```

### Grafana Dashboards

```bash
# Acceder a Grafana:
http://localhost:3000

# Credenciales por defecto:
# Usuario: admin
# Password: admin
```

### Health Checks

```bash
# Verificar salud completa
curl http://localhost:8000/health

# Verificar componentes específicos
curl http://localhost:8000/health/mt5
curl http://localhost:8000/health/database
curl http://localhost:8000/health/redis
```

## 🧪 Testing

### Tests Unitarios

```bash
pytest tests/test_suite.py -v
```

### Tests de Integración

```bash
pytest tests/test_integration.py -v
```

### Tests de Performance

```bash
pytest tests/test_performance.py -v
```

### Tests de Carga

```bash
python tests/load_testing.py --concurrent-users 100 --duration 300
```

## 📚 Documentación

### Documentación Completa

La documentación completa está disponible en `docs/`:

- **Arquitectura**: `docs/architecture.md`
- **API Reference**: `docs/api_reference.md`
- **Guía de Usuario**: `docs/user_guide.md`
- **Guía de Desarrollo**: `docs/development_guide.md`

### Generar Documentación

```bash
# Documentación API con Sphinx
cd docs
make html

# Abrir documentación
open _build/html/index.html
```

## 🔍 Troubleshooting

### Problema: MT5 no se conecta

**Solución**:
```bash
# 1. Verificar que MT5 esté instalado y corriendo
# 2. Verificar credenciales en config/platform_config.yaml
# 3. Verificar firewall y antivirus
# 4. Intentar conexión manual:
python -c "import MetaTrader5 as mt5; print(mt5.initialize())"
```

### Problema: Error de base de datos

**Solución**:
```bash
# 1. Verificar que PostgreSQL esté corriendo
systemctl status postgresql

# 2. Verificar conexión
psql -U trading_user -d trading -h localhost

# 3. Recrear tablas si es necesario
python -c "from database.data_manager import TradingData; TradingData.create_tables()"
```

### Problema: Memory leaks

**Solución**:
```bash
# 1. Verificar uso de memoria
python -c "from core.performance_optimizer import PerformanceOptimizer; p = PerformanceOptimizer(); print(p.memory_usage_report())"

# 2. Limpiar cache
python -c "from core.platform import get_platform; p = get_platform(); p.data_manager.redis_client.flushdb()"

# 3. Reiniciar servicios
python main.py --restart
```

## 🤝 Contribuir

### Proceso de Contribución

1. Fork el repositorio
2. Crear branch de feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

### Estándares de Código

- **Style Guide**: PEP 8
- **Docstrings**: Google style
- **Type Hints**: Obligatorios en funciones públicas
- **Tests**: Coverage mínimo 80%

### Ejecutar Linters

```bash
# Black (formateo)
black . --check

# Flake8 (linting)
flake8 . --max-line-length=100

# MyPy (type checking)
mypy . --ignore-missing-imports

# Todo junto
./scripts/lint.sh
```

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👥 Autores

- **Tu Nombre** - *Trabajo inicial* - [tu-github](https://github.com/tu-usuario)

## 🙏 Agradecimientos

- Comunidad de Python y trading algorítmico
- Contribuidores de librerías de código abierto
- MetaQuotes por MetaTrader 5 API

## 📞 Soporte

- **Email**: support@trading-platform.com
- **Discord**: [Unirse al servidor](https://discord.gg/trading-platform)
- **Issues**: [GitHub Issues](https://github.com/tu-usuario/trading-platform/issues)

## 🗺️ Roadmap

### Versión 1.1 (Q1 2025)
- [ ] Soporte para Binance y otros exchanges
- [ ] Estrategias de arbitraje
- [ ] Panel de control web (Streamlit/Dash)
- [ ] Mobile app (React Native)

### Versión 1.2 (Q2 2025)
- [ ] Integración con TradingView
- [ ] Social trading features
- [ ] Automated strategy marketplace
- [ ] Advanced portfolio optimization

### Versión 2.0 (Q3 2025)
- [ ] Deep Learning con PyTorch
- [ ] Reinforcement Learning avanzado
- [ ] Sentiment analysis con NLP
- [ ] Quantum computing integration (experimental)

## ⚠️ Disclaimer

**ADVERTENCIA**: El trading algorítmico conlleva riesgos significativos. Esta plataforma es una herramienta y no garantiza ganancias. Siempre opera con capital que puedes permitirte perder. Los resultados pasados no garantizan rendimientos futuros.

El uso de esta plataforma es bajo tu propio riesgo. Los desarrolladores no son responsables de pérdidas incurridas.

---

**¿Preguntas?** Abre un issue o únete a nuestra comunidad en Discord.

**¿Te gusta el proyecto?** Dale una ⭐ en GitHub y comparte con otros traders!