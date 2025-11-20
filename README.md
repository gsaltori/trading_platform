# Plataforma de Trading Algorítmico Avanzada

Sistema completo de trading algorítmico con Machine Learning, optimización inteligente y ejecución en vivo para MetaTrader 5.

## 🚀 Características Principales

### 🤖 Machine Learning Avanzado
- **Múltiples algoritmos**: XGBoost, Random Forest, LSTM, Ensemble
- **Feature engineering automático**: 50+ indicadores técnicos
- **Detección de regímenes** de mercado
- **Modelos predictivos** para dirección de precios

### 🔧 Optimización Inteligente
- **Algoritmos genéticos** con NSGA-II
- **Optimización bayesiana**
- **Multi-objetivo** (Sharpe vs Drawdown)
- **Paralelización** masiva

### 📊 Backtesting de Alta Performance
- **Vectorizado** con Numba
- **Ejecución realista** (slippage, spread variable)
- **Métricas avanzadas** (Sharpe, Sortino, Calmar, Omega)
- **Walk-forward analysis**

### ⚡ Ejecución en Vivo
- **Conexión nativa MT5**
- **Gestión automática** de órdenes
- **Risk management** en tiempo real
- **Circuit breakers** inteligentes

### 🎨 Interfaz Moderna
- **Dashboard** en tiempo real
- **Editor visual** de estrategias
- **Gráficos interactivos** con Plotly
- **Modo oscuro/claro**

## 🛠 Instalación Rápida

### Requisitos
- Python 3.10+
- MetaTrader 5 instalado
- 8GB RAM mínimo (16GB recomendado)

### Instalación
```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/trading-platform.git
cd trading-platform

# Instalar dependencias
pip install -r requirements.txt

# Configurar entorno
python main.py --deploy --environment development