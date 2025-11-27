# Trading Platform - GUI de Generación de Estrategias

## 🚀 Inicio Rápido

### Instalación de Dependencias

**Windows (método rápido):**
```batch
install_dependencies.bat
```

**Manual:**
```bash
pip install pandas numpy pyyaml joblib scikit-learn ta PyQt6 matplotlib
```

**Opcional - MetaTrader 5:**
```bash
pip install MetaTrader5
```

**Opcional - ML avanzado:**
```bash
pip install xgboost lightgbm
```

### Ejecutar la GUI

**Windows:**
```batch
start_gui.bat
```

**O directamente con Python:**
```bash
python run_gui.py
```

---

## 📖 Guía de Uso

### 1. Conexión a MetaTrader 5

1. En el panel izquierdo, haz clic en **"🔍 Scan"** para detectar instalaciones de MT5
2. Selecciona una instalación de la lista
3. Ingresa tus credenciales (servidor, login, password)
4. Haz clic en **"🔌 Connect"**

### 2. Descarga de Datos

1. Ve a la pestaña **"📥 Data"**
2. Selecciona un preset de símbolos o agrega manualmente
3. Configura el timeframe y rango de fechas
4. Haz clic en **"📥 Download Data"**

### 3. Generación de Estrategias

#### Estrategias Basadas en Reglas:
1. Ve a la pestaña **"🔧 Strategy Generator"** → **"📊 Rule-Based Strategies"**
2. Selecciona el tipo de estrategia (MA Crossover, RSI, MACD)
3. Configura los parámetros
4. Haz clic en **"🔧 Generate Strategy"**

#### Estrategias con ML:
1. Ve a **"🤖 ML Strategies"**
2. Selecciona el algoritmo (Random Forest, XGBoost, etc.)
3. Configura los parámetros de ML
4. Haz clic en **"🤖 Train ML Model"**

#### Optimización:
1. Ve a **"⚡ Optimization"**
2. Configura los rangos de parámetros
3. Selecciona la métrica de optimización
4. Haz clic en **"⚡ Run Optimization"**

### 4. Backtesting

1. Ve a la pestaña **"📊 Backtest"**
2. Configura comisiones y slippage
3. Haz clic en **"▶️ Run Backtest"**
4. Revisa los resultados y métricas

### 5. Análisis de Gráficos

1. Ve a la pestaña **"📈 Charts"**
2. Selecciona el tipo de gráfico
3. Activa/desactiva señales e indicadores
4. Usa la curva de equity para análisis de rendimiento

---

## 🏗️ Estructura del Proyecto

```
trading_platform/
├── ui/
│   ├── main_window.py          # Ventana principal
│   ├── widgets/
│   │   ├── mt5_connection_widget.py   # Conexión MT5
│   │   ├── data_panel.py              # Panel de datos
│   │   ├── strategy_generator_widget.py # Generador de estrategias
│   │   ├── backtest_results_widget.py # Resultados de backtest
│   │   ├── charts_widget.py           # Gráficos
│   │   └── log_widget.py              # Logs
│   └── utils/
│       ├── mt5_discovery.py    # Descubrimiento de MT5
│       └── workers.py          # Workers en segundo plano
├── data/
│   └── mt5_connector.py        # Conector MT5
├── strategies/
│   └── strategy_engine.py      # Motor de estrategias
├── ml/
│   └── ml_engine.py            # Motor de ML
├── backtesting/
│   └── backtest_engine.py      # Motor de backtesting
├── config/
│   └── settings.py             # Configuración
├── run_gui.py                  # Lanzador de GUI
├── start_gui.bat               # Script de inicio (Windows)
└── install_dependencies.bat    # Instalador de dependencias
```

---

## 📊 Características

### Conexión MT5
- ✅ Detección automática de instalaciones
- ✅ Soporte para múltiples cuentas
- ✅ Información de cuenta en tiempo real
- ✅ Lista de símbolos disponibles

### Gestión de Datos
- ✅ Descarga de datos históricos
- ✅ Múltiples timeframes (M1-MN1)
- ✅ Múltiples símbolos simultáneos
- ✅ Exportación a CSV

### Generación de Estrategias
- ✅ Moving Average Crossover
- ✅ RSI con divergencias
- ✅ MACD
- ✅ Machine Learning (Random Forest, XGBoost, etc.)
- ✅ Optimización de parámetros

### Backtesting
- ✅ Simulación realista con slippage
- ✅ Comisiones configurables
- ✅ Métricas avanzadas (Sharpe, Sortino, etc.)
- ✅ Análisis de trades
- ✅ Exportación de reportes

### Visualización
- ✅ Gráficos de velas
- ✅ Indicadores técnicos
- ✅ Señales de trading
- ✅ Curva de equity
- ✅ Drawdown

---

## ⚙️ Requisitos del Sistema

- **Python:** 3.10 o superior
- **Sistema Operativo:** Windows 10/11 (para MT5)
- **RAM:** 4GB mínimo, 8GB recomendado
- **MetaTrader 5:** Instalado y con cuenta activa

---

## 🐛 Solución de Problemas

### "PyQt6 no está instalado"
```bash
pip install PyQt6
```

### "No se puede conectar a MT5"
1. Verifica que MT5 esté instalado y funcionando
2. Asegúrate de que la cuenta tenga permiso de trading algorítmico
3. Verifica las credenciales

### "No hay datos disponibles"
1. Verifica la conexión a MT5
2. Comprueba que el símbolo existe en tu broker
3. Intenta con un rango de fechas diferente

### Los gráficos no se muestran
```bash
pip install matplotlib
```

---

## 📝 Licencia

Este proyecto es para uso educativo y personal.

---

## 🤝 Soporte

Para problemas o sugerencias, utiliza el botón de feedback en la aplicación.
