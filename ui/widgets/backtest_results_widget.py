# ui/widgets/backtest_results_widget.py
"""
Backtest Results Widget.

Displays comprehensive backtest results and metrics.
"""

import logging
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QTextEdit, QSplitter, QFrame,
    QGridLayout, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThreadPool
from PyQt6.QtGui import QFont, QColor

logger = logging.getLogger(__name__)


class MetricCard(QFrame):
    """A card widget for displaying a single metric."""
    
    def __init__(self, title: str, value: str = "-", color: str = None, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Segoe UI", 9))
        self.title_label.setStyleSheet("color: #6c757d;")
        
        self.value_label = QLabel(value)
        self.value_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        if color:
            self.value_label.setStyleSheet(f"color: {color};")
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
    
    def set_value(self, value: str, color: str = None):
        """Update the metric value."""
        self.value_label.setText(value)
        if color:
            self.value_label.setStyleSheet(f"color: {color};")
        else:
            self.value_label.setStyleSheet("")


class BacktestResultsWidget(QWidget):
    """Widget for displaying backtest results."""
    
    # Signals
    backtest_requested = pyqtSignal(object, object)  # strategy, data
    
    def __init__(self, backtest_engine=None, parent=None):
        super().__init__(parent)
        self.backtest_engine = backtest_engine
        self.current_result = None
        self.thread_pool = QThreadPool()
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Backtest controls
        controls_group = QGroupBox("Backtest Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        controls_layout.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.setMinimumWidth(100)
        controls_layout.addWidget(self.symbol_combo)
        
        controls_layout.addWidget(QLabel("Commission:"))
        self.commission_combo = QComboBox()
        self.commission_combo.addItems(["0.01%", "0.05%", "0.1%", "0.2%"])
        self.commission_combo.setCurrentIndex(2)
        controls_layout.addWidget(self.commission_combo)
        
        controls_layout.addWidget(QLabel("Slippage:"))
        self.slippage_combo = QComboBox()
        self.slippage_combo.addItems(["Fixed", "Dynamic"])
        controls_layout.addWidget(self.slippage_combo)
        
        self.run_backtest_btn = QPushButton("▶️ Run Backtest")
        self.run_backtest_btn.setMinimumWidth(120)
        self.run_backtest_btn.clicked.connect(self.request_backtest)
        controls_layout.addWidget(self.run_backtest_btn)
        
        controls_layout.addStretch()
        layout.addWidget(controls_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Metrics summary
        metrics_group = QGroupBox("Performance Summary")
        metrics_layout = QGridLayout(metrics_group)
        
        self.total_return_card = MetricCard("Total Return")
        metrics_layout.addWidget(self.total_return_card, 0, 0)
        
        self.sharpe_card = MetricCard("Sharpe Ratio")
        metrics_layout.addWidget(self.sharpe_card, 0, 1)
        
        self.max_dd_card = MetricCard("Max Drawdown")
        metrics_layout.addWidget(self.max_dd_card, 0, 2)
        
        self.win_rate_card = MetricCard("Win Rate")
        metrics_layout.addWidget(self.win_rate_card, 0, 3)
        
        self.profit_factor_card = MetricCard("Profit Factor")
        metrics_layout.addWidget(self.profit_factor_card, 1, 0)
        
        self.total_trades_card = MetricCard("Total Trades")
        metrics_layout.addWidget(self.total_trades_card, 1, 1)
        
        self.avg_trade_card = MetricCard("Avg Trade")
        metrics_layout.addWidget(self.avg_trade_card, 1, 2)
        
        self.calmar_card = MetricCard("Calmar Ratio")
        metrics_layout.addWidget(self.calmar_card, 1, 3)
        
        layout.addWidget(metrics_group)
        
        # Detailed results tabs
        tabs = QTabWidget()
        
        # Trades tab
        trades_tab = self.create_trades_tab()
        tabs.addTab(trades_tab, "📋 Trades")
        
        # Statistics tab
        stats_tab = self.create_statistics_tab()
        tabs.addTab(stats_tab, "📊 Statistics")
        
        # Monthly returns tab
        monthly_tab = self.create_monthly_tab()
        tabs.addTab(monthly_tab, "📅 Monthly")
        
        layout.addWidget(tabs)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_trades_btn = QPushButton("📤 Export Trades")
        self.export_trades_btn.clicked.connect(self.export_trades)
        self.export_trades_btn.setEnabled(False)
        export_layout.addWidget(self.export_trades_btn)
        
        self.export_report_btn = QPushButton("📄 Export Report")
        self.export_report_btn.clicked.connect(self.export_report)
        self.export_report_btn.setEnabled(False)
        export_layout.addWidget(self.export_report_btn)
        
        export_layout.addStretch()
        layout.addLayout(export_layout)
    
    def create_trades_tab(self) -> QWidget:
        """Create trades table tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(10)
        self.trades_table.setHorizontalHeaderLabels([
            "Entry Time", "Exit Time", "Symbol", "Direction",
            "Entry Price", "Exit Price", "Volume", "P&L", "P&L %", "Exit Reason"
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trades_table.setAlternatingRowColors(True)
        
        layout.addWidget(self.trades_table)
        return widget
    
    def create_statistics_tab(self) -> QWidget:
        """Create detailed statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Consolas", 10))
        
        layout.addWidget(self.stats_text)
        return widget
    
    def create_monthly_tab(self) -> QWidget:
        """Create monthly returns tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.monthly_table = QTableWidget()
        self.monthly_table.setAlternatingRowColors(True)
        
        layout.addWidget(self.monthly_table)
        return widget
    
    def set_symbols(self, symbols: List[str]):
        """Set available symbols."""
        self.symbol_combo.clear()
        self.symbol_combo.addItems(symbols)
    
    def request_backtest(self):
        """Request a backtest run."""
        self.backtest_requested.emit(None, None)
    
    def run_backtest(self, strategy, data: pd.DataFrame, symbol: str = None):
        """Run backtest with given strategy and data."""
        if self.backtest_engine is None:
            QMessageBox.warning(self, "Error", "Backtest engine not initialized")
            return
        
        self.run_backtest_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Get commission
        commission_text = self.commission_combo.currentText()
        commission = float(commission_text.replace('%', '')) / 100
        
        # Get slippage model
        slippage_model = self.slippage_combo.currentText().lower()
        
        symbol = symbol or self.symbol_combo.currentText() or "SYMBOL"
        
        from ..utils.workers import BacktestWorker
        
        worker = BacktestWorker(
            self.backtest_engine,
            strategy,
            data,
            symbol,
            commission,
            slippage_model
        )
        
        worker.signals.progress.connect(self.on_progress)
        worker.signals.result.connect(self.display_results)
        worker.signals.error.connect(self.on_error)
        
        self.thread_pool.start(worker)
    
    def on_progress(self, percentage: int, message: str):
        """Handle progress update."""
        self.progress_bar.setValue(percentage)
    
    def on_error(self, error: str, traceback: str):
        """Handle backtest error."""
        self.run_backtest_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Backtest Error", error)
    
    def display_results(self, result):
        """Display backtest results."""
        self.run_backtest_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.current_result = result
        
        # Update metric cards
        total_return = result.total_return
        return_color = "green" if total_return > 0 else "red" if total_return < 0 else None
        self.total_return_card.set_value(f"{total_return:+.2f}%", return_color)
        
        sharpe = result.sharpe_ratio
        sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "N/A"
        sharpe_color = "green" if sharpe and sharpe > 1 else "orange" if sharpe and sharpe > 0 else "red" if sharpe else None
        self.sharpe_card.set_value(sharpe_str, sharpe_color)
        
        max_dd = result.max_drawdown
        dd_color = "green" if max_dd < 10 else "orange" if max_dd < 20 else "red"
        self.max_dd_card.set_value(f"{max_dd:.2f}%", dd_color)
        
        win_rate = result.win_rate
        wr_color = "green" if win_rate > 50 else "orange" if win_rate > 40 else "red"
        self.win_rate_card.set_value(f"{win_rate:.1f}%", wr_color)
        
        profit_factor = result.profit_factor
        pf_str = f"{profit_factor:.2f}" if profit_factor and profit_factor != float('inf') else "∞" if profit_factor == float('inf') else "N/A"
        pf_color = "green" if profit_factor and profit_factor > 1.5 else "orange" if profit_factor and profit_factor > 1 else "red" if profit_factor else None
        self.profit_factor_card.set_value(pf_str, pf_color)
        
        self.total_trades_card.set_value(str(result.total_trades))
        
        avg_trade = result.avg_trade
        avg_color = "green" if avg_trade > 0 else "red" if avg_trade < 0 else None
        self.avg_trade_card.set_value(f"${avg_trade:+.2f}", avg_color)
        
        calmar = result.calmar_ratio
        calmar_str = f"{calmar:.2f}" if calmar is not None else "N/A"
        self.calmar_card.set_value(calmar_str)
        
        # Update trades table
        self.populate_trades_table(result.trades)
        
        # Update statistics
        self.update_statistics(result)
        
        # Update monthly returns
        self.update_monthly_returns(result)
        
        # Enable export
        self.export_trades_btn.setEnabled(True)
        self.export_report_btn.setEnabled(True)
    
    def populate_trades_table(self, trades: List):
        """Populate trades table."""
        self.trades_table.setRowCount(len(trades))
        
        for i, trade in enumerate(trades):
            self.trades_table.setItem(i, 0, QTableWidgetItem(str(trade.entry_time)))
            self.trades_table.setItem(i, 1, QTableWidgetItem(str(trade.exit_time or "-")))
            self.trades_table.setItem(i, 2, QTableWidgetItem(trade.symbol))
            
            direction = "🔼 LONG" if trade.direction > 0 else "🔽 SHORT"
            dir_item = QTableWidgetItem(direction)
            dir_item.setForeground(QColor("green" if trade.direction > 0 else "red"))
            self.trades_table.setItem(i, 3, dir_item)
            
            self.trades_table.setItem(i, 4, QTableWidgetItem(f"{trade.entry_price:.5f}"))
            self.trades_table.setItem(i, 5, QTableWidgetItem(f"{trade.exit_price:.5f}" if trade.exit_price else "-"))
            self.trades_table.setItem(i, 6, QTableWidgetItem(f"{trade.volume:.2f}"))
            
            pnl = trade.pnl or 0
            pnl_item = QTableWidgetItem(f"${pnl:+.2f}")
            pnl_item.setForeground(QColor("green" if pnl > 0 else "red" if pnl < 0 else "black"))
            self.trades_table.setItem(i, 7, pnl_item)
            
            pnl_pct = trade.pnl_pct or 0
            pnl_pct_item = QTableWidgetItem(f"{pnl_pct:+.2f}%")
            pnl_pct_item.setForeground(QColor("green" if pnl_pct > 0 else "red" if pnl_pct < 0 else "black"))
            self.trades_table.setItem(i, 8, pnl_pct_item)
            
            exit_reason = trade.metadata.get('exit_reason', '-')
            self.trades_table.setItem(i, 9, QTableWidgetItem(exit_reason))
    
    def update_statistics(self, result):
        """Update detailed statistics."""
        stats = f"""
╔══════════════════════════════════════════════════════════════╗
║                    BACKTEST REPORT                          ║
╠══════════════════════════════════════════════════════════════╣
║  Strategy: {result.strategy_name:<48} ║
╠══════════════════════════════════════════════════════════════╣
║  PERFORMANCE METRICS                                         ║
╠══════════════════════════════════════════════════════════════╣
║  Total Return:        {result.total_return:>+10.2f}%                           ║
║  Sharpe Ratio:        {result.sharpe_ratio if result.sharpe_ratio else 'N/A':>10}                           ║
║  Sortino Ratio:       {result.sortino_ratio if result.sortino_ratio else 'N/A':>10}                           ║
║  Calmar Ratio:        {result.calmar_ratio if result.calmar_ratio else 'N/A':>10}                           ║
║  Max Drawdown:        {result.max_drawdown:>10.2f}%                           ║
║  Profit Factor:       {result.profit_factor if result.profit_factor and result.profit_factor != float('inf') else 'N/A':>10}                           ║
║  Recovery Factor:     {result.recovery_factor if result.recovery_factor else 'N/A':>10}                           ║
╠══════════════════════════════════════════════════════════════╣
║  TRADE STATISTICS                                            ║
╠══════════════════════════════════════════════════════════════╣
║  Total Trades:        {result.total_trades:>10}                           ║
║  Winning Trades:      {result.winning_trades:>10}                           ║
║  Losing Trades:       {result.losing_trades:>10}                           ║
║  Win Rate:            {result.win_rate:>10.1f}%                           ║
║  Average Trade:       ${result.avg_trade:>+9.2f}                           ║
║  Avg Winning Trade:   ${result.avg_winning_trade:>+9.2f}                           ║
║  Avg Losing Trade:    ${result.avg_losing_trade:>+9.2f}                           ║
╚══════════════════════════════════════════════════════════════╝
"""
        self.stats_text.setText(stats)
    
    def update_monthly_returns(self, result):
        """Update monthly returns table."""
        if result.equity_curve.empty:
            return
        
        try:
            # Calculate monthly returns from equity curve
            equity = result.equity_curve.set_index('timestamp')['equity']
            monthly = equity.resample('M').last().pct_change() * 100
            
            # Create pivot table by year/month
            years = sorted(set(monthly.index.year))
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            self.monthly_table.setRowCount(len(years))
            self.monthly_table.setColumnCount(13)  # 12 months + YTD
            self.monthly_table.setHorizontalHeaderLabels(months + ['YTD'])
            self.monthly_table.setVerticalHeaderLabels([str(y) for y in years])
            
            for row, year in enumerate(years):
                year_data = monthly[monthly.index.year == year]
                ytd = 0
                
                for month in range(1, 13):
                    try:
                        value = year_data[year_data.index.month == month].iloc[0]
                        item = QTableWidgetItem(f"{value:+.1f}%")
                        item.setForeground(QColor("green" if value > 0 else "red" if value < 0 else "black"))
                        self.monthly_table.setItem(row, month - 1, item)
                        ytd += value
                    except (IndexError, KeyError):
                        self.monthly_table.setItem(row, month - 1, QTableWidgetItem("-"))
                
                # YTD
                ytd_item = QTableWidgetItem(f"{ytd:+.1f}%")
                ytd_item.setForeground(QColor("green" if ytd > 0 else "red" if ytd < 0 else "black"))
                ytd_item.setFont(QFont("", -1, QFont.Weight.Bold))
                self.monthly_table.setItem(row, 12, ytd_item)
                
        except Exception as e:
            logger.warning(f"Error calculating monthly returns: {e}")
    
    def export_trades(self):
        """Export trades to CSV."""
        if not self.current_result or not self.current_result.trades:
            return
        
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Trades",
            f"trades_{self.current_result.strategy_name}.csv",
            "CSV Files (*.csv)"
        )
        
        if filename:
            trades_data = []
            for trade in self.current_result.trades:
                trades_data.append({
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'symbol': trade.symbol,
                    'direction': 'LONG' if trade.direction > 0 else 'SHORT',
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'volume': trade.volume,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'exit_reason': trade.metadata.get('exit_reason', '')
                })
            
            df = pd.DataFrame(trades_data)
            df.to_csv(filename, index=False)
            QMessageBox.information(self, "Export Complete", f"Trades exported to {filename}")
    
    def export_report(self):
        """Export full report."""
        if not self.current_result:
            return
        
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Report",
            f"report_{self.current_result.strategy_name}.html",
            "HTML Files (*.html)"
        )
        
        if filename:
            self.generate_html_report(filename)
            QMessageBox.information(self, "Export Complete", f"Report exported to {filename}")
    
    def generate_html_report(self, filename: str):
        """Generate an HTML report."""
        result = self.current_result
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report - {result.strategy_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
        th {{ background: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Backtest Report</h1>
        <h2>{result.strategy_name}</h2>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-title">Total Return</div>
            <div class="metric-value {'positive' if result.total_return > 0 else 'negative'}">{result.total_return:+.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Sharpe Ratio</div>
            <div class="metric-value">{result.sharpe_ratio:.2f if result.sharpe_ratio else 'N/A'}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Max Drawdown</div>
            <div class="metric-value negative">{result.max_drawdown:.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Win Rate</div>
            <div class="metric-value">{result.win_rate:.1f}%</div>
        </div>
    </div>
    
    <h3>Trade Statistics</h3>
    <table>
        <tr><td>Total Trades</td><td>{result.total_trades}</td></tr>
        <tr><td>Winning Trades</td><td>{result.winning_trades}</td></tr>
        <tr><td>Losing Trades</td><td>{result.losing_trades}</td></tr>
        <tr><td>Average Trade</td><td>${result.avg_trade:+.2f}</td></tr>
        <tr><td>Profit Factor</td><td>{result.profit_factor:.2f if result.profit_factor and result.profit_factor != float('inf') else 'N/A'}</td></tr>
    </table>
    
    <h3>Trades</h3>
    <table>
        <tr>
            <th>Entry Time</th>
            <th>Exit Time</th>
            <th>Direction</th>
            <th>Entry Price</th>
            <th>Exit Price</th>
            <th>P&L</th>
        </tr>
"""
        
        for trade in result.trades[:50]:  # Limit to 50 trades
            pnl_class = 'positive' if (trade.pnl or 0) > 0 else 'negative'
            html += f"""
        <tr>
            <td>{trade.entry_time}</td>
            <td>{trade.exit_time or '-'}</td>
            <td>{'LONG' if trade.direction > 0 else 'SHORT'}</td>
            <td>{trade.entry_price:.5f}</td>
            <td>{trade.exit_price:.5f if trade.exit_price else '-'}</td>
            <td class="{pnl_class}">${trade.pnl or 0:+.2f}</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        
        with open(filename, 'w') as f:
            f.write(html)
