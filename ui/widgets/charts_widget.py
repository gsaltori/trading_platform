# ui/widgets/charts_widget.py
"""
Charts Widget.

Provides interactive charts for price data and equity curves.
"""

import logging
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QCheckBox, QSlider, QSpinBox,
    QTabWidget, QSplitter, QToolBar
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont

logger = logging.getLogger(__name__)

# Try to import matplotlib
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available for charts")


class CandlestickChart(QWidget):
    """Custom candlestick chart widget using matplotlib."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.DataFrame] = None
        
        if MATPLOTLIB_AVAILABLE:
            self.setup_matplotlib_ui()
        else:
            self.setup_fallback_ui()
    
    def setup_matplotlib_ui(self):
        layout = QVBoxLayout(self)
        
        # Create figure
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
    
    def setup_fallback_ui(self):
        layout = QVBoxLayout(self)
        label = QLabel("Matplotlib not available. Install with: pip install matplotlib")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
    
    def plot_data(self, data: pd.DataFrame, signals: pd.DataFrame = None, 
                  indicators: List[str] = None):
        """Plot candlestick chart with optional signals and indicators."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self.data = data
        self.signals = signals
        
        self.figure.clear()
        
        # Create subplots
        if 'volume' in data.columns:
            gs = self.figure.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.1)
            ax_price = self.figure.add_subplot(gs[0])
            ax_volume = self.figure.add_subplot(gs[1], sharex=ax_price)
            ax_rsi = self.figure.add_subplot(gs[2], sharex=ax_price)
            ax_macd = self.figure.add_subplot(gs[3], sharex=ax_price)
        else:
            gs = self.figure.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
            ax_price = self.figure.add_subplot(gs[0])
            ax_volume = None
            ax_rsi = self.figure.add_subplot(gs[1], sharex=ax_price)
            ax_macd = self.figure.add_subplot(gs[2], sharex=ax_price)
        
        # Plot candlesticks
        self.plot_candlesticks(ax_price, data)
        
        # Plot moving averages if available
        if 'ma_fast' in data.columns:
            ax_price.plot(data.index, data['ma_fast'], label='Fast MA', alpha=0.7)
        if 'ma_slow' in data.columns:
            ax_price.plot(data.index, data['ma_slow'], label='Slow MA', alpha=0.7)
        
        # Plot signals
        if signals is not None and 'signal' in signals.columns:
            buy_signals = signals[signals['signal'] > 0]
            sell_signals = signals[signals['signal'] < 0]
            
            ax_price.scatter(buy_signals.index, buy_signals['close'] * 0.998,
                           marker='^', color='green', s=100, label='Buy', zorder=5)
            ax_price.scatter(sell_signals.index, sell_signals['close'] * 1.002,
                           marker='v', color='red', s=100, label='Sell', zorder=5)
        
        ax_price.set_ylabel('Price')
        ax_price.legend(loc='upper left')
        ax_price.grid(True, alpha=0.3)
        
        # Plot volume
        if ax_volume is not None and 'volume' in data.columns:
            colors = ['green' if data['close'].iloc[i] >= data['open'].iloc[i] 
                     else 'red' for i in range(len(data))]
            ax_volume.bar(data.index, data['volume'], color=colors, alpha=0.5, width=0.8)
            ax_volume.set_ylabel('Volume')
            ax_volume.grid(True, alpha=0.3)
        
        # Plot RSI
        if 'rsi' in data.columns:
            ax_rsi.plot(data.index, data['rsi'], color='purple', label='RSI')
            ax_rsi.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax_rsi.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax_rsi.fill_between(data.index, 30, 70, alpha=0.1)
            ax_rsi.set_ylabel('RSI')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.grid(True, alpha=0.3)
        
        # Plot MACD
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            ax_macd.plot(data.index, data['macd'], label='MACD', color='blue')
            ax_macd.plot(data.index, data['macd_signal'], label='Signal', color='orange')
            if 'macd_histogram' in data.columns:
                colors = ['green' if v >= 0 else 'red' for v in data['macd_histogram']]
                ax_macd.bar(data.index, data['macd_histogram'], color=colors, alpha=0.5, width=0.8)
            ax_macd.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax_macd.set_ylabel('MACD')
            ax_macd.legend(loc='upper left')
            ax_macd.grid(True, alpha=0.3)
        
        # Format x-axis
        ax_macd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_macd.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        self.figure.autofmt_xdate()
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_candlesticks(self, ax, data: pd.DataFrame):
        """Plot candlestick bars."""
        width = 0.6
        width2 = 0.1
        
        up = data[data['close'] >= data['open']]
        down = data[data['close'] < data['open']]
        
        # Plot up candles
        ax.bar(up.index, up['close'] - up['open'], width, bottom=up['open'], color='green', alpha=0.8)
        ax.bar(up.index, up['high'] - up['close'], width2, bottom=up['close'], color='green', alpha=0.8)
        ax.bar(up.index, up['low'] - up['open'], width2, bottom=up['open'], color='green', alpha=0.8)
        
        # Plot down candles
        ax.bar(down.index, down['close'] - down['open'], width, bottom=down['open'], color='red', alpha=0.8)
        ax.bar(down.index, down['high'] - down['open'], width2, bottom=down['open'], color='red', alpha=0.8)
        ax.bar(down.index, down['low'] - down['close'], width2, bottom=down['close'], color='red', alpha=0.8)


class EquityCurveChart(QWidget):
    """Equity curve chart widget."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        if MATPLOTLIB_AVAILABLE:
            self.setup_matplotlib_ui()
        else:
            self.setup_fallback_ui()
    
    def setup_matplotlib_ui(self):
        layout = QVBoxLayout(self)
        
        self.figure = Figure(figsize=(10, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
    
    def setup_fallback_ui(self):
        layout = QVBoxLayout(self)
        label = QLabel("Matplotlib not available")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
    
    def plot_equity_curve(self, equity_df: pd.DataFrame, initial_capital: float = 10000):
        """Plot equity curve with drawdown."""
        if not MATPLOTLIB_AVAILABLE or equity_df.empty:
            return
        
        self.figure.clear()
        
        gs = self.figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
        ax_equity = self.figure.add_subplot(gs[0])
        ax_dd = self.figure.add_subplot(gs[1], sharex=ax_equity)
        
        # Plot equity
        ax_equity.plot(equity_df['timestamp'], equity_df['equity'], 
                      color='blue', linewidth=1.5, label='Equity')
        ax_equity.axhline(y=initial_capital, color='gray', linestyle='--', 
                         alpha=0.5, label='Initial Capital')
        
        # Fill above/below initial
        ax_equity.fill_between(equity_df['timestamp'], initial_capital, equity_df['equity'],
                              where=equity_df['equity'] >= initial_capital,
                              color='green', alpha=0.3)
        ax_equity.fill_between(equity_df['timestamp'], initial_capital, equity_df['equity'],
                              where=equity_df['equity'] < initial_capital,
                              color='red', alpha=0.3)
        
        ax_equity.set_ylabel('Equity ($)')
        ax_equity.legend(loc='upper left')
        ax_equity.grid(True, alpha=0.3)
        
        # Plot drawdown
        if 'drawdown_pct' in equity_df.columns:
            ax_dd.fill_between(equity_df['timestamp'], 0, equity_df['drawdown_pct'],
                              color='red', alpha=0.5)
            ax_dd.set_ylabel('Drawdown (%)')
            ax_dd.set_ylim(equity_df['drawdown_pct'].min() * 1.1, 0)
            ax_dd.grid(True, alpha=0.3)
        
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        self.figure.autofmt_xdate()
        self.figure.tight_layout()
        self.canvas.draw()


class ChartsWidget(QWidget):
    """Main charts widget with multiple chart types."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Chart controls
        controls = QHBoxLayout()
        
        controls.addWidget(QLabel("Chart Type:"))
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["Candlestick", "Line", "OHLC"])
        self.chart_type_combo.currentTextChanged.connect(self.on_chart_type_changed)
        controls.addWidget(self.chart_type_combo)
        
        controls.addWidget(QLabel("Timeframe:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["All", "1 Month", "3 Months", "6 Months", "1 Year"])
        self.timeframe_combo.currentTextChanged.connect(self.on_timeframe_changed)
        controls.addWidget(self.timeframe_combo)
        
        self.show_signals_check = QCheckBox("Show Signals")
        self.show_signals_check.setChecked(True)
        self.show_signals_check.toggled.connect(self.refresh_chart)
        controls.addWidget(self.show_signals_check)
        
        self.show_indicators_check = QCheckBox("Show Indicators")
        self.show_indicators_check.setChecked(True)
        self.show_indicators_check.toggled.connect(self.refresh_chart)
        controls.addWidget(self.show_indicators_check)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # Tab widget for different charts
        self.tabs = QTabWidget()
        
        # Price chart tab
        self.candlestick_chart = CandlestickChart()
        self.tabs.addTab(self.candlestick_chart, "📈 Price Chart")
        
        # Equity curve tab
        self.equity_chart = EquityCurveChart()
        self.tabs.addTab(self.equity_chart, "💰 Equity Curve")
        
        layout.addWidget(self.tabs)
        
        # Store data references
        self.current_data = None
        self.current_signals = None
        self.current_equity = None
    
    def on_chart_type_changed(self, chart_type: str):
        """Handle chart type change."""
        self.refresh_chart()
    
    def on_timeframe_changed(self, timeframe: str):
        """Handle timeframe change."""
        self.refresh_chart()
    
    def refresh_chart(self):
        """Refresh the current chart."""
        if self.current_data is not None:
            self.plot_price_data(self.current_data, self.current_signals)
    
    def plot_price_data(self, data: pd.DataFrame, signals: pd.DataFrame = None):
        """Plot price data on the candlestick chart."""
        self.current_data = data
        self.current_signals = signals if self.show_signals_check.isChecked() else None
        
        # Filter by timeframe
        timeframe = self.timeframe_combo.currentText()
        filtered_data = self.filter_by_timeframe(data, timeframe)
        filtered_signals = None
        if signals is not None and self.show_signals_check.isChecked():
            filtered_signals = signals.loc[signals.index.isin(filtered_data.index)]
        
        self.candlestick_chart.plot_data(filtered_data, filtered_signals)
    
    def plot_equity_curve(self, equity_df: pd.DataFrame, initial_capital: float = 10000):
        """Plot equity curve."""
        self.current_equity = equity_df
        self.equity_chart.plot_equity_curve(equity_df, initial_capital)
    
    def filter_by_timeframe(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Filter data by selected timeframe."""
        if timeframe == "All":
            return data
        
        from datetime import timedelta
        
        end_date = data.index[-1]
        
        if timeframe == "1 Month":
            start_date = end_date - timedelta(days=30)
        elif timeframe == "3 Months":
            start_date = end_date - timedelta(days=90)
        elif timeframe == "6 Months":
            start_date = end_date - timedelta(days=180)
        elif timeframe == "1 Year":
            start_date = end_date - timedelta(days=365)
        else:
            return data
        
        return data[data.index >= start_date]
    
    def clear_charts(self):
        """Clear all charts."""
        self.current_data = None
        self.current_signals = None
        self.current_equity = None
        
        if MATPLOTLIB_AVAILABLE:
            self.candlestick_chart.figure.clear()
            self.candlestick_chart.canvas.draw()
            self.equity_chart.figure.clear()
            self.equity_chart.canvas.draw()
