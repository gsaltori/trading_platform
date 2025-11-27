# ui/widgets/strategy_generator_widget.py
"""
Strategy Generator Widget.

Automatic strategy generation using ML and optimization.
"""

import logging
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QLineEdit, QFormLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QProgressBar, QTabWidget,
    QTextEdit, QSplitter, QListWidget, QListWidgetItem,
    QTreeWidget, QTreeWidgetItem, QDialog, QDialogButtonBox,
    QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QThreadPool
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)


class ParameterRangeDialog(QDialog):
    """Dialog for configuring parameter optimization ranges."""
    
    def __init__(self, strategy_type: str, parent=None):
        super().__init__(parent)
        self.strategy_type = strategy_type
        self.setWindowTitle(f"Optimization Parameters - {strategy_type}")
        self.setMinimumWidth(500)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Parameter table
        self.param_table = QTableWidget()
        self.param_table.setColumnCount(4)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Min", "Max", "Step"])
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Default parameters based on strategy type
        params = self.get_default_params()
        
        self.param_table.setRowCount(len(params))
        for i, (name, (min_val, max_val, step)) in enumerate(params.items()):
            self.param_table.setItem(i, 0, QTableWidgetItem(name))
            
            min_spin = QSpinBox() if isinstance(min_val, int) else QDoubleSpinBox()
            min_spin.setRange(-10000, 10000)
            min_spin.setValue(min_val)
            self.param_table.setCellWidget(i, 1, min_spin)
            
            max_spin = QSpinBox() if isinstance(max_val, int) else QDoubleSpinBox()
            max_spin.setRange(-10000, 10000)
            max_spin.setValue(max_val)
            self.param_table.setCellWidget(i, 2, max_spin)
            
            step_spin = QSpinBox() if isinstance(step, int) else QDoubleSpinBox()
            step_spin.setRange(1, 1000)
            step_spin.setValue(step)
            self.param_table.setCellWidget(i, 3, step_spin)
        
        layout.addWidget(self.param_table)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_default_params(self) -> Dict[str, tuple]:
        """Get default parameter ranges for strategy type."""
        if self.strategy_type == 'ma_crossover':
            return {
                'fast_period': (5, 20, 5),
                'slow_period': (20, 50, 10),
                'rsi_period': (7, 21, 7),
                'rsi_oversold': (20, 40, 10),
                'rsi_overbought': (60, 80, 10)
            }
        elif self.strategy_type == 'rsi':
            return {
                'rsi_period': (7, 21, 7),
                'rsi_oversold': (20, 40, 5),
                'rsi_overbought': (60, 80, 5),
                'ma_period': (14, 28, 7)
            }
        elif self.strategy_type == 'macd':
            return {
                'fast_period': (8, 16, 4),
                'slow_period': (20, 32, 4),
                'signal_period': (6, 12, 3)
            }
        else:
            return {}
    
    def get_ranges(self) -> Dict[str, List]:
        """Get parameter ranges as lists for optimization."""
        ranges = {}
        
        for i in range(self.param_table.rowCount()):
            name = self.param_table.item(i, 0).text()
            min_widget = self.param_table.cellWidget(i, 1)
            max_widget = self.param_table.cellWidget(i, 2)
            step_widget = self.param_table.cellWidget(i, 3)
            
            min_val = min_widget.value()
            max_val = max_widget.value()
            step = step_widget.value()
            
            if isinstance(min_val, int):
                ranges[name] = list(range(int(min_val), int(max_val) + 1, int(step)))
            else:
                ranges[name] = list(np.arange(min_val, max_val + step, step))
        
        return ranges


class StrategyGeneratorWidget(QWidget):
    """Widget for automatic strategy generation."""
    
    # Signals
    strategy_generated = pyqtSignal(str, object)  # name, strategy
    strategy_optimized = pyqtSignal(str, dict)  # name, best_params
    ml_model_trained = pyqtSignal(str, dict)  # name, results
    
    STRATEGY_TYPES = {
        'ma_crossover': 'Moving Average Crossover',
        'rsi': 'RSI Strategy',
        'macd': 'MACD Strategy'
    }
    
    ML_ALGORITHMS = {
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'svm': 'Support Vector Machine',
        'mlp': 'Neural Network (MLP)',
        'lstm': 'LSTM (Deep Learning)',
        'ensemble': 'Ensemble (Multiple Models)'
    }
    
    OPTIMIZATION_METRICS = {
        'sharpe_ratio': 'Sharpe Ratio',
        'total_return': 'Total Return',
        'win_rate': 'Win Rate',
        'profit_factor': 'Profit Factor'
    }
    
    def __init__(self, strategy_engine=None, ml_engine=None, parent=None):
        super().__init__(parent)
        self.strategy_engine = strategy_engine
        self.ml_engine = ml_engine
        self.data: Optional[pd.DataFrame] = None
        self.thread_pool = QThreadPool()
        self.generated_strategies = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create tabs
        tabs = QTabWidget()
        
        # Tab 1: Rule-Based Strategy Generation
        rule_tab = self.create_rule_based_tab()
        tabs.addTab(rule_tab, "📊 Rule-Based Strategies")
        
        # Tab 2: ML-Based Strategy Generation
        ml_tab = self.create_ml_tab()
        tabs.addTab(ml_tab, "🤖 ML Strategies")
        
        # Tab 3: Strategy Optimization
        opt_tab = self.create_optimization_tab()
        tabs.addTab(opt_tab, "⚡ Optimization")
        
        # Tab 4: Generated Strategies
        gen_tab = self.create_generated_tab()
        tabs.addTab(gen_tab, "📋 Generated Strategies")
        
        layout.addWidget(tabs)
    
    def create_rule_based_tab(self) -> QWidget:
        """Create rule-based strategy generation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Strategy Type Selection
        type_group = QGroupBox("Strategy Type")
        type_layout = QFormLayout(type_group)
        
        self.strategy_type_combo = QComboBox()
        for key, name in self.STRATEGY_TYPES.items():
            self.strategy_type_combo.addItem(name, key)
        self.strategy_type_combo.currentIndexChanged.connect(self.on_strategy_type_changed)
        type_layout.addRow("Strategy:", self.strategy_type_combo)
        
        self.strategy_name_edit = QLineEdit()
        self.strategy_name_edit.setPlaceholderText("Enter strategy name")
        type_layout.addRow("Name:", self.strategy_name_edit)
        
        layout.addWidget(type_group)
        
        # Parameters
        params_group = QGroupBox("Strategy Parameters")
        self.params_layout = QFormLayout(params_group)
        
        # Will be populated based on strategy type
        self.param_widgets = {}
        self.populate_strategy_params()
        
        layout.addWidget(params_group)
        
        # Risk Management
        risk_group = QGroupBox("Risk Management")
        risk_layout = QFormLayout(risk_group)
        
        self.atr_mult_spin = QDoubleSpinBox()
        self.atr_mult_spin.setRange(0.5, 5.0)
        self.atr_mult_spin.setValue(2.0)
        self.atr_mult_spin.setSingleStep(0.5)
        risk_layout.addRow("ATR Multiplier (SL):", self.atr_mult_spin)
        
        self.rr_ratio_spin = QDoubleSpinBox()
        self.rr_ratio_spin.setRange(0.5, 5.0)
        self.rr_ratio_spin.setValue(1.5)
        self.rr_ratio_spin.setSingleStep(0.5)
        risk_layout.addRow("Risk/Reward Ratio:", self.rr_ratio_spin)
        
        layout.addWidget(risk_group)
        
        # Generate button
        self.generate_rule_btn = QPushButton("🔧 Generate Strategy")
        self.generate_rule_btn.setMinimumHeight(40)
        self.generate_rule_btn.clicked.connect(self.generate_rule_based_strategy)
        layout.addWidget(self.generate_rule_btn)
        
        layout.addStretch()
        return widget
    
    def create_ml_tab(self) -> QWidget:
        """Create ML-based strategy generation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # ML Algorithm Selection
        algo_group = QGroupBox("ML Algorithm")
        algo_layout = QFormLayout(algo_group)
        
        self.ml_algo_combo = QComboBox()
        for key, name in self.ML_ALGORITHMS.items():
            self.ml_algo_combo.addItem(name, key)
        algo_layout.addRow("Algorithm:", self.ml_algo_combo)
        
        self.ml_name_edit = QLineEdit()
        self.ml_name_edit.setPlaceholderText("ML strategy name")
        algo_layout.addRow("Name:", self.ml_name_edit)
        
        layout.addWidget(algo_group)
        
        # ML Parameters
        ml_params_group = QGroupBox("ML Parameters")
        ml_params_layout = QFormLayout(ml_params_group)
        
        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(10, 200)
        self.lookback_spin.setValue(50)
        ml_params_layout.addRow("Lookback Window:", self.lookback_spin)
        
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 20)
        self.horizon_spin.setValue(1)
        ml_params_layout.addRow("Prediction Horizon:", self.horizon_spin)
        
        self.train_split_spin = QDoubleSpinBox()
        self.train_split_spin.setRange(0.5, 0.9)
        self.train_split_spin.setValue(0.8)
        self.train_split_spin.setSingleStep(0.05)
        ml_params_layout.addRow("Train/Test Split:", self.train_split_spin)
        
        layout.addWidget(ml_params_group)
        
        # Feature Selection
        feature_group = QGroupBox("Feature Engineering")
        feature_layout = QVBoxLayout(feature_group)
        
        self.use_ta_features = QCheckBox("Technical Analysis Features")
        self.use_ta_features.setChecked(True)
        feature_layout.addWidget(self.use_ta_features)
        
        self.use_price_features = QCheckBox("Price Pattern Features")
        self.use_price_features.setChecked(True)
        feature_layout.addWidget(self.use_price_features)
        
        self.use_volume_features = QCheckBox("Volume Features")
        self.use_volume_features.setChecked(True)
        feature_layout.addWidget(self.use_volume_features)
        
        layout.addWidget(feature_group)
        
        # Train button and progress
        self.train_ml_btn = QPushButton("🤖 Train ML Model")
        self.train_ml_btn.setMinimumHeight(40)
        self.train_ml_btn.clicked.connect(self.train_ml_model)
        layout.addWidget(self.train_ml_btn)
        
        self.ml_progress = QProgressBar()
        self.ml_progress.setVisible(False)
        layout.addWidget(self.ml_progress)
        
        # Results
        results_group = QGroupBox("Training Results")
        results_layout = QVBoxLayout(results_group)
        
        self.ml_results_text = QTextEdit()
        self.ml_results_text.setReadOnly(True)
        self.ml_results_text.setMaximumHeight(200)
        results_layout.addWidget(self.ml_results_text)
        
        layout.addWidget(results_group)
        
        layout.addStretch()
        return widget
    
    def create_optimization_tab(self) -> QWidget:
        """Create strategy optimization tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Strategy to optimize
        select_group = QGroupBox("Strategy Selection")
        select_layout = QFormLayout(select_group)
        
        self.opt_strategy_combo = QComboBox()
        for key, name in self.STRATEGY_TYPES.items():
            self.opt_strategy_combo.addItem(name, key)
        select_layout.addRow("Strategy Type:", self.opt_strategy_combo)
        
        self.opt_metric_combo = QComboBox()
        for key, name in self.OPTIMIZATION_METRICS.items():
            self.opt_metric_combo.addItem(name, key)
        select_layout.addRow("Optimization Metric:", self.opt_metric_combo)
        
        layout.addWidget(select_group)
        
        # Parameter ranges button
        self.config_params_btn = QPushButton("⚙️ Configure Parameter Ranges")
        self.config_params_btn.clicked.connect(self.configure_parameter_ranges)
        layout.addWidget(self.config_params_btn)
        
        # Parameter ranges display
        self.param_ranges_text = QTextEdit()
        self.param_ranges_text.setReadOnly(True)
        self.param_ranges_text.setMaximumHeight(150)
        self.param_ranges_text.setPlaceholderText("Click 'Configure Parameter Ranges' to set optimization parameters")
        layout.addWidget(self.param_ranges_text)
        
        # Optimization settings
        opt_settings_group = QGroupBox("Optimization Settings")
        opt_settings_layout = QFormLayout(opt_settings_group)
        
        self.use_genetic = QCheckBox("Use Genetic Algorithm")
        self.use_genetic.setChecked(False)
        opt_settings_layout.addRow("", self.use_genetic)
        
        self.population_spin = QSpinBox()
        self.population_spin.setRange(20, 500)
        self.population_spin.setValue(100)
        opt_settings_layout.addRow("Population Size:", self.population_spin)
        
        self.generations_spin = QSpinBox()
        self.generations_spin.setRange(10, 200)
        self.generations_spin.setValue(50)
        opt_settings_layout.addRow("Generations:", self.generations_spin)
        
        layout.addWidget(opt_settings_group)
        
        # Optimize button
        self.optimize_btn = QPushButton("⚡ Run Optimization")
        self.optimize_btn.setMinimumHeight(40)
        self.optimize_btn.clicked.connect(self.run_optimization)
        layout.addWidget(self.optimize_btn)
        
        self.opt_progress = QProgressBar()
        self.opt_progress.setVisible(False)
        layout.addWidget(self.opt_progress)
        
        # Optimization results
        opt_results_group = QGroupBox("Optimization Results")
        opt_results_layout = QVBoxLayout(opt_results_group)
        
        self.opt_results_text = QTextEdit()
        self.opt_results_text.setReadOnly(True)
        opt_results_layout.addWidget(self.opt_results_text)
        
        layout.addWidget(opt_results_group)
        
        return widget
    
    def create_generated_tab(self) -> QWidget:
        """Create tab showing generated strategies."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Strategies list
        self.strategies_tree = QTreeWidget()
        self.strategies_tree.setHeaderLabels(["Strategy", "Type", "Parameters", "Status"])
        self.strategies_tree.setColumnWidth(0, 200)
        self.strategies_tree.setColumnWidth(1, 100)
        self.strategies_tree.setColumnWidth(2, 200)
        self.strategies_tree.itemClicked.connect(self.on_strategy_selected)
        layout.addWidget(self.strategies_tree)
        
        # Strategy details
        details_group = QGroupBox("Strategy Details")
        details_layout = QVBoxLayout(details_group)
        
        self.strategy_details_text = QTextEdit()
        self.strategy_details_text.setReadOnly(True)
        details_layout.addWidget(self.strategy_details_text)
        
        layout.addWidget(details_group)
        
        # Actions
        actions_layout = QHBoxLayout()
        
        self.backtest_strategy_btn = QPushButton("📊 Backtest")
        self.backtest_strategy_btn.clicked.connect(self.backtest_selected_strategy)
        self.backtest_strategy_btn.setEnabled(False)
        actions_layout.addWidget(self.backtest_strategy_btn)
        
        self.export_strategy_btn = QPushButton("📤 Export")
        self.export_strategy_btn.clicked.connect(self.export_selected_strategy)
        self.export_strategy_btn.setEnabled(False)
        actions_layout.addWidget(self.export_strategy_btn)
        
        self.delete_strategy_btn = QPushButton("🗑️ Delete")
        self.delete_strategy_btn.clicked.connect(self.delete_selected_strategy)
        self.delete_strategy_btn.setEnabled(False)
        actions_layout.addWidget(self.delete_strategy_btn)
        
        layout.addLayout(actions_layout)
        
        return widget
    
    def populate_strategy_params(self):
        """Populate parameter widgets based on strategy type."""
        # Clear existing widgets
        while self.params_layout.rowCount() > 0:
            self.params_layout.removeRow(0)
        self.param_widgets.clear()
        
        strategy_type = self.strategy_type_combo.currentData()
        
        if strategy_type == 'ma_crossover':
            self.add_param_widget("fast_period", "Fast MA Period:", 5, 100, 10)
            self.add_param_widget("slow_period", "Slow MA Period:", 10, 200, 20)
            
            ma_type_combo = QComboBox()
            ma_type_combo.addItems(["sma", "ema"])
            self.params_layout.addRow("MA Type:", ma_type_combo)
            self.param_widgets["ma_type"] = ma_type_combo
            
            self.add_param_widget("rsi_period", "RSI Period:", 5, 30, 14)
            self.add_param_widget("rsi_oversold", "RSI Oversold:", 10, 40, 30)
            self.add_param_widget("rsi_overbought", "RSI Overbought:", 60, 90, 70)
            
        elif strategy_type == 'rsi':
            self.add_param_widget("rsi_period", "RSI Period:", 5, 30, 14)
            self.add_param_widget("rsi_oversold", "RSI Oversold:", 10, 40, 30)
            self.add_param_widget("rsi_overbought", "RSI Overbought:", 60, 90, 70)
            self.add_param_widget("ma_period", "MA Period:", 10, 50, 21)
            
            use_div = QCheckBox()
            use_div.setChecked(True)
            self.params_layout.addRow("Use Divergences:", use_div)
            self.param_widgets["use_divergences"] = use_div
            
        elif strategy_type == 'macd':
            self.add_param_widget("fast_period", "Fast Period:", 8, 20, 12)
            self.add_param_widget("slow_period", "Slow Period:", 20, 35, 26)
            self.add_param_widget("signal_period", "Signal Period:", 5, 15, 9)
    
    def add_param_widget(self, name: str, label: str, min_val: int, max_val: int, default: int):
        """Add a parameter spinbox widget."""
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        self.params_layout.addRow(label, spin)
        self.param_widgets[name] = spin
    
    def on_strategy_type_changed(self, index: int):
        """Handle strategy type change."""
        self.populate_strategy_params()
    
    def set_data(self, data: pd.DataFrame):
        """Set the data for strategy generation."""
        self.data = data
    
    def generate_rule_based_strategy(self):
        """Generate a rule-based strategy."""
        if self.strategy_engine is None:
            QMessageBox.warning(self, "Error", "Strategy engine not initialized")
            return
        
        strategy_type = self.strategy_type_combo.currentData()
        strategy_name = self.strategy_name_edit.text().strip()
        
        if not strategy_name:
            strategy_name = f"{strategy_type}_{len(self.generated_strategies) + 1}"
        
        # Collect parameters
        params = {}
        for name, widget in self.param_widgets.items():
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                params[name] = widget.isChecked()
        
        # Risk management
        risk_management = {
            'atr_multiplier': self.atr_mult_spin.value(),
            'risk_reward_ratio': self.rr_ratio_spin.value()
        }
        
        try:
            from strategies.strategy_engine import StrategyConfig
            
            config = StrategyConfig(
                name=strategy_name,
                symbols=[],  # Will be set later
                timeframe="H1",
                parameters=params,
                risk_management=risk_management
            )
            
            strategy = self.strategy_engine.create_strategy(strategy_type, config)
            
            self.generated_strategies[strategy_name] = {
                'type': 'rule_based',
                'strategy_type': strategy_type,
                'strategy': strategy,
                'config': config,
                'params': params
            }
            
            self.update_strategies_tree()
            self.strategy_generated.emit(strategy_name, strategy)
            
            QMessageBox.information(self, "Success", f"Strategy '{strategy_name}' generated successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating strategy: {e}")
            logger.exception("Strategy generation error")
    
    def train_ml_model(self):
        """Train an ML model for strategy generation."""
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please load data first")
            return
        
        if self.ml_engine is None:
            QMessageBox.warning(self, "Error", "ML engine not initialized")
            return
        
        algorithm = self.ml_algo_combo.currentData()
        name = self.ml_name_edit.text().strip()
        
        if not name:
            name = f"ml_{algorithm}_{len(self.generated_strategies) + 1}"
        
        self.train_ml_btn.setEnabled(False)
        self.ml_progress.setVisible(True)
        self.ml_progress.setValue(0)
        self.ml_results_text.clear()
        
        from ..utils.workers import StrategyGeneratorWorker
        
        worker = StrategyGeneratorWorker(
            self.ml_engine,
            self.data.copy(),
            target_type='classification',
            algorithms=[algorithm],
            optimize=True
        )
        
        worker.signals.progress.connect(self.on_ml_progress)
        worker.signals.result.connect(lambda r: self.on_ml_complete(r, name, algorithm))
        worker.signals.error.connect(self.on_ml_error)
        worker.signals.log.connect(lambda l, m: logger.log(getattr(logging, l, logging.INFO), m))
        
        self.thread_pool.start(worker)
    
    def on_ml_progress(self, percentage: int, message: str):
        """Handle ML training progress."""
        self.ml_progress.setValue(percentage)
    
    def on_ml_complete(self, results: Dict, name: str, algorithm: str):
        """Handle ML training completion."""
        self.train_ml_btn.setEnabled(True)
        self.ml_progress.setVisible(False)
        
        # Display results
        results_text = f"=== ML Training Results ===\n\n"
        
        for algo, res in results.get('results', {}).items():
            results_text += f"Algorithm: {algo}\n"
            if 'error' in res:
                results_text += f"  Error: {res['error']}\n"
            else:
                results_text += f"  Accuracy: {res['metrics'].get('accuracy', 0):.4f}\n"
                results_text += f"  Precision: {res['metrics'].get('precision', 0):.4f}\n"
                results_text += f"  Recall: {res['metrics'].get('recall', 0):.4f}\n"
                results_text += f"  F1 Score: {res['metrics'].get('f1', 0):.4f}\n"
            results_text += "\n"
        
        results_text += f"Best Model: {results.get('best_model', 'N/A')}\n"
        results_text += f"Best Accuracy: {results.get('best_accuracy', 0):.4f}\n"
        
        self.ml_results_text.setText(results_text)
        
        # Save strategy
        self.generated_strategies[name] = {
            'type': 'ml',
            'algorithm': algorithm,
            'results': results,
            'model_name': results.get('best_model')
        }
        
        self.update_strategies_tree()
        self.ml_model_trained.emit(name, results)
    
    def on_ml_error(self, error: str, traceback: str):
        """Handle ML training error."""
        self.train_ml_btn.setEnabled(True)
        self.ml_progress.setVisible(False)
        self.ml_results_text.setText(f"Error: {error}\n\n{traceback}")
        QMessageBox.critical(self, "ML Training Error", error)
    
    def configure_parameter_ranges(self):
        """Show dialog to configure parameter ranges."""
        strategy_type = self.opt_strategy_combo.currentData()
        dialog = ParameterRangeDialog(strategy_type, self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.current_param_ranges = dialog.get_ranges()
            
            # Display ranges
            text = "Parameter Ranges:\n\n"
            for name, values in self.current_param_ranges.items():
                text += f"  {name}: {min(values)} to {max(values)} ({len(values)} values)\n"
            
            total_combinations = 1
            for values in self.current_param_ranges.values():
                total_combinations *= len(values)
            text += f"\nTotal combinations to test: {total_combinations:,}"
            
            self.param_ranges_text.setText(text)
    
    def run_optimization(self):
        """Run strategy optimization."""
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please load data first")
            return
        
        if not hasattr(self, 'current_param_ranges') or not self.current_param_ranges:
            QMessageBox.warning(self, "No Parameters", "Please configure parameter ranges first")
            return
        
        strategy_type = self.opt_strategy_combo.currentData()
        metric = self.opt_metric_combo.currentData()
        
        self.optimize_btn.setEnabled(False)
        self.opt_progress.setVisible(True)
        self.opt_progress.setValue(0)
        self.opt_results_text.clear()
        
        from ..utils.workers import OptimizationWorker
        
        worker = OptimizationWorker(
            self.strategy_engine,
            strategy_type,
            self.data.copy(),
            self.current_param_ranges,
            metric,
            callback=self.on_optimization_iteration
        )
        
        worker.signals.progress.connect(self.on_optimization_progress)
        worker.signals.result.connect(lambda r: self.on_optimization_complete(r, strategy_type))
        worker.signals.error.connect(self.on_optimization_error)
        
        self.thread_pool.start(worker)
    
    def on_optimization_iteration(self, iteration: int, result: Dict):
        """Handle optimization iteration."""
        pass  # Could update a real-time chart here
    
    def on_optimization_progress(self, percentage: int, message: str):
        """Handle optimization progress."""
        self.opt_progress.setValue(percentage)
    
    def on_optimization_complete(self, results: Dict, strategy_type: str):
        """Handle optimization completion."""
        self.optimize_btn.setEnabled(True)
        self.opt_progress.setVisible(False)
        
        best_params = results.get('best_params', {})
        best_metric = results.get('best_metric', 0)
        
        # Display results
        text = "=== Optimization Results ===\n\n"
        text += f"Best {self.opt_metric_combo.currentText()}: {best_metric:.4f}\n\n"
        text += "Optimal Parameters:\n"
        for name, value in best_params.items():
            text += f"  {name}: {value}\n"
        
        text += f"\nTotal combinations tested: {len(results.get('all_results', []))}"
        
        self.opt_results_text.setText(text)
        
        # Create optimized strategy
        opt_name = f"optimized_{strategy_type}_{len(self.generated_strategies) + 1}"
        
        try:
            from strategies.strategy_engine import StrategyConfig
            
            config = StrategyConfig(
                name=opt_name,
                symbols=[],
                timeframe="H1",
                parameters=best_params
            )
            
            strategy = self.strategy_engine.create_strategy(strategy_type, config)
            
            self.generated_strategies[opt_name] = {
                'type': 'optimized',
                'strategy_type': strategy_type,
                'strategy': strategy,
                'config': config,
                'params': best_params,
                'metric': best_metric
            }
            
            self.update_strategies_tree()
            self.strategy_optimized.emit(opt_name, best_params)
            
            QMessageBox.information(
                self, "Optimization Complete",
                f"Optimized strategy '{opt_name}' created with {self.opt_metric_combo.currentText()}: {best_metric:.4f}"
            )
            
        except Exception as e:
            logger.exception("Error creating optimized strategy")
    
    def on_optimization_error(self, error: str, traceback: str):
        """Handle optimization error."""
        self.optimize_btn.setEnabled(True)
        self.opt_progress.setVisible(False)
        self.opt_results_text.setText(f"Error: {error}\n\n{traceback}")
        QMessageBox.critical(self, "Optimization Error", error)
    
    def update_strategies_tree(self):
        """Update the strategies tree view."""
        self.strategies_tree.clear()
        
        for name, data in self.generated_strategies.items():
            item = QTreeWidgetItem()
            item.setText(0, name)
            item.setText(1, data.get('type', 'unknown'))
            
            if data['type'] == 'rule_based':
                params_str = ", ".join(f"{k}={v}" for k, v in list(data.get('params', {}).items())[:3])
                item.setText(2, params_str)
            elif data['type'] == 'ml':
                item.setText(2, f"Algorithm: {data.get('algorithm', 'unknown')}")
            elif data['type'] == 'optimized':
                metric = data.get('metric', 0)
                item.setText(2, f"Metric: {metric:.4f}")
            
            item.setText(3, "Ready")
            item.setData(0, Qt.ItemDataRole.UserRole, name)
            
            self.strategies_tree.addTopLevelItem(item)
    
    def on_strategy_selected(self, item: QTreeWidgetItem, column: int):
        """Handle strategy selection."""
        name = item.data(0, Qt.ItemDataRole.UserRole)
        data = self.generated_strategies.get(name)
        
        if data:
            details = f"=== {name} ===\n\n"
            details += f"Type: {data.get('type', 'unknown')}\n"
            
            if data['type'] == 'rule_based':
                details += f"Strategy Type: {data.get('strategy_type', 'unknown')}\n\n"
                details += "Parameters:\n"
                for k, v in data.get('params', {}).items():
                    details += f"  {k}: {v}\n"
                
            elif data['type'] == 'ml':
                details += f"Algorithm: {data.get('algorithm', 'unknown')}\n"
                results = data.get('results', {})
                if results:
                    details += f"Best Accuracy: {results.get('best_accuracy', 0):.4f}\n"
                    
            elif data['type'] == 'optimized':
                details += f"Strategy Type: {data.get('strategy_type', 'unknown')}\n"
                details += f"Optimized Metric: {data.get('metric', 0):.4f}\n\n"
                details += "Best Parameters:\n"
                for k, v in data.get('params', {}).items():
                    details += f"  {k}: {v}\n"
            
            self.strategy_details_text.setText(details)
            
            self.backtest_strategy_btn.setEnabled(True)
            self.export_strategy_btn.setEnabled(True)
            self.delete_strategy_btn.setEnabled(True)
    
    def backtest_selected_strategy(self):
        """Backtest the selected strategy."""
        current = self.strategies_tree.currentItem()
        if current:
            name = current.data(0, Qt.ItemDataRole.UserRole)
            data = self.generated_strategies.get(name)
            if data and 'strategy' in data:
                self.strategy_generated.emit(name, data['strategy'])
    
    def export_selected_strategy(self):
        """Export the selected strategy."""
        current = self.strategies_tree.currentItem()
        if not current:
            return
        
        name = current.data(0, Qt.ItemDataRole.UserRole)
        data = self.generated_strategies.get(name)
        
        if not data:
            return
        
        from PyQt6.QtWidgets import QFileDialog
        import json
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Strategy",
            f"{name}.json",
            "JSON Files (*.json)"
        )
        
        if filename:
            export_data = {
                'name': name,
                'type': data.get('type'),
                'strategy_type': data.get('strategy_type'),
                'params': data.get('params', {}),
                'metric': data.get('metric')
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            QMessageBox.information(self, "Export Complete", f"Strategy exported to {filename}")
    
    def delete_selected_strategy(self):
        """Delete the selected strategy."""
        current = self.strategies_tree.currentItem()
        if not current:
            return
        
        name = current.data(0, Qt.ItemDataRole.UserRole)
        
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete strategy '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if name in self.generated_strategies:
                del self.generated_strategies[name]
            self.update_strategies_tree()
            self.strategy_details_text.clear()
            
            self.backtest_strategy_btn.setEnabled(False)
            self.export_strategy_btn.setEnabled(False)
            self.delete_strategy_btn.setEnabled(False)
    
    def get_strategy(self, name: str) -> Optional[Any]:
        """Get a generated strategy by name."""
        data = self.generated_strategies.get(name)
        if data:
            return data.get('strategy')
        return None
