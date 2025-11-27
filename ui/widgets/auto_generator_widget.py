# ui/widgets/auto_generator_widget.py
"""
Advanced Auto Strategy Generator Widget.

Provides a complete interface for automatic strategy generation,
genetic optimization, validation, and MQL5 export.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Check PyQt6 availability
PYQT_AVAILABLE = False
try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
        QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
        QProgressBar, QTextEdit, QTabWidget, QTableWidget, QTableWidgetItem,
        QTreeWidget, QTreeWidgetItem, QSplitter, QFileDialog, QMessageBox,
        QListWidget, QListWidgetItem, QFrame, QScrollArea, QGridLayout,
        QHeaderView, QAbstractItemView
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QThreadPool, QTimer
    from PyQt6.QtGui import QFont, QColor
    PYQT_AVAILABLE = True
except ImportError:
    logger.warning("PyQt6 not available")


if PYQT_AVAILABLE:
    from ..utils.workers import BaseWorker, WorkerSignals
    
    
    class GeneticOptimizationWorker(BaseWorker):
        """Worker for genetic optimization."""
        
        def __init__(self, optimizer, data, backtest_engine, symbols, timeframe):
            super().__init__()
            self.optimizer = optimizer
            self.data = data
            self.backtest_engine = backtest_engine
            self.symbols = symbols
            self.timeframe = timeframe
        
        def execute(self):
            def progress_callback(generation, stats):
                if not self._is_cancelled:
                    progress = int((generation + 1) / self.optimizer.config.generations * 100)
                    self.emit_progress(progress, f"Gen {generation + 1}: Best={stats['best_fitness']:.4f}")
            
            best_strategy = self.optimizer.run(
                data=self.data,
                backtest_engine=self.backtest_engine,
                symbols=self.symbols,
                timeframe=self.timeframe,
                callback=progress_callback
            )
            
            return {
                'best_strategy': best_strategy,
                'top_strategies': self.optimizer.get_top_strategies(10),
                'history': self.optimizer.generation_history
            }
    
    
    class WalkForwardWorker(BaseWorker):
        """Worker for walk-forward analysis."""
        
        def __init__(self, analyzer, data, strategy_class, strategy_config, backtest_engine):
            super().__init__()
            self.analyzer = analyzer
            self.data = data
            self.strategy_class = strategy_class
            self.strategy_config = strategy_config
            self.backtest_engine = backtest_engine
        
        def execute(self):
            def progress_callback(current, total, result):
                if not self._is_cancelled:
                    progress = int(current / total * 100)
                    self.emit_progress(progress, f"Window {current}/{total}")
            
            report = self.analyzer.run_analysis(
                data=self.data,
                strategy_class=self.strategy_class,
                strategy_config=self.strategy_config,
                backtest_engine=self.backtest_engine,
                callback=progress_callback
            )
            
            return report
    
    
    class AutoGeneratorWidget(QWidget):
        """
        Advanced widget for automatic strategy generation.
        
        Features:
        - Random strategy generation
        - Genetic algorithm optimization
        - Walk-forward validation
        - Overfitting detection
        - MQL5 export
        """
        
        # Signals
        strategy_generated = pyqtSignal(str, object)  # name, strategy
        strategies_optimized = pyqtSignal(list)  # list of strategies
        validation_complete = pyqtSignal(dict)  # validation results
        
        def __init__(self, parent=None):
            super().__init__(parent)
            
            # Initialize components
            self.thread_pool = QThreadPool()
            self.current_data = None
            self.generated_strategies: Dict[str, Any] = {}
            self.best_strategy = None
            self._stop_requested = False  # Flag for stopping generation
            
            # Initialize engines (lazy loading)
            self._auto_generator = None
            self._genetic_optimizer = None
            self._walk_forward = None
            self._overfit_detector = None
            self._mql5_exporter = None
            self._backtest_engine = None
            
            self.setup_ui()
        
        @property
        def auto_generator(self):
            if self._auto_generator is None:
                # Use extended generator with all 77 indicators
                from strategies.indicator_integration import create_extended_generator
                self._auto_generator = create_extended_generator()
            return self._auto_generator
        
        @property
        def genetic_optimizer(self):
            if self._genetic_optimizer is None:
                from strategies.genetic_strategy import GeneticStrategyOptimizer, GeneticConfig
                config = GeneticConfig(
                    population_size=self.population_spin.value(),
                    generations=self.generations_spin.value(),
                    mutation_rate=self.mutation_spin.value(),
                    crossover_rate=self.crossover_spin.value()
                )
                self._genetic_optimizer = GeneticStrategyOptimizer(config)
            return self._genetic_optimizer
        
        @property
        def walk_forward(self):
            if self._walk_forward is None:
                from optimization.walk_forward import WalkForwardAnalyzer
                self._walk_forward = WalkForwardAnalyzer({
                    'num_windows': self.wf_windows_spin.value(),
                    'train_ratio': self.wf_train_ratio_spin.value()
                })
            return self._walk_forward
        
        @property
        def overfit_detector(self):
            if self._overfit_detector is None:
                from optimization.overfitting_detector import OverfittingDetector
                self._overfit_detector = OverfittingDetector()
            return self._overfit_detector
        
        @property
        def mql5_exporter(self):
            if self._mql5_exporter is None:
                from strategies.mql5_exporter import MQL5Exporter
                self._mql5_exporter = MQL5Exporter()
            return self._mql5_exporter
        
        @property
        def backtest_engine(self):
            if self._backtest_engine is None:
                from backtesting.backtest_engine import BacktestEngine
                self._backtest_engine = BacktestEngine(initial_capital=10000)
            return self._backtest_engine
        
        def setup_ui(self):
            """Setup the user interface."""
            layout = QVBoxLayout(self)
            layout.setContentsMargins(5, 5, 5, 5)
            
            # Main splitter
            splitter = QSplitter(Qt.Orientation.Horizontal)
            
            # Left panel - Controls
            left_panel = self._create_controls_panel()
            splitter.addWidget(left_panel)
            
            # Right panel - Results
            right_panel = self._create_results_panel()
            splitter.addWidget(right_panel)
            
            splitter.setSizes([400, 600])
            layout.addWidget(splitter)
        
        def _create_controls_panel(self) -> QWidget:
            """Create the controls panel."""
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # Scroll area for controls
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)
            
            # === Generation Settings ===
            gen_group = QGroupBox("🎲 Random Generation")
            gen_layout = QVBoxLayout(gen_group)
            
            # Number of strategies
            row = QHBoxLayout()
            row.addWidget(QLabel("Strategies to generate:"))
            self.num_strategies_spin = QSpinBox()
            self.num_strategies_spin.setRange(1, 100)
            self.num_strategies_spin.setValue(10)
            row.addWidget(self.num_strategies_spin)
            gen_layout.addLayout(row)
            
            # Timeframe
            row = QHBoxLayout()
            row.addWidget(QLabel("Timeframe:"))
            self.timeframe_combo = QComboBox()
            self.timeframe_combo.addItems(['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1'])
            self.timeframe_combo.setCurrentText('H1')
            row.addWidget(self.timeframe_combo)
            gen_layout.addLayout(row)
            
            # Symbol
            row = QHBoxLayout()
            row.addWidget(QLabel("Symbol:"))
            self.symbol_combo = QComboBox()
            self.symbol_combo.setEditable(True)
            self.symbol_combo.addItems(['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'BTCUSD'])
            row.addWidget(self.symbol_combo)
            gen_layout.addLayout(row)
            
            # Indicator settings
            row = QHBoxLayout()
            row.addWidget(QLabel("Min indicators:"))
            self.min_indicators_spin = QSpinBox()
            self.min_indicators_spin.setRange(1, 10)
            self.min_indicators_spin.setValue(2)
            row.addWidget(self.min_indicators_spin)
            row.addWidget(QLabel("Max:"))
            self.max_indicators_spin = QSpinBox()
            self.max_indicators_spin.setRange(1, 10)
            self.max_indicators_spin.setValue(5)
            row.addWidget(self.max_indicators_spin)
            gen_layout.addLayout(row)
            
            # Generate button
            self.generate_btn = QPushButton("🎲 Generate Random Strategies")
            self.generate_btn.clicked.connect(self.generate_random_strategies)
            self.generate_btn.setStyleSheet("QPushButton { padding: 10px; font-weight: bold; }")
            gen_layout.addWidget(self.generate_btn)
            
            scroll_layout.addWidget(gen_group)
            
            # === Profitability Filters ===
            filter_group = QGroupBox("🎯 Profitability Filters")
            filter_layout = QVBoxLayout(filter_group)
            
            # Enable filtering
            self.enable_filters_check = QCheckBox("Only keep profitable strategies")
            self.enable_filters_check.setChecked(True)
            filter_layout.addWidget(self.enable_filters_check)
            
            # Minimum net profit %
            row = QHBoxLayout()
            row.addWidget(QLabel("Min Net Profit %:"))
            self.min_profit_spin = QDoubleSpinBox()
            self.min_profit_spin.setRange(-100, 1000)
            self.min_profit_spin.setValue(0)  # Must be positive
            self.min_profit_spin.setSuffix("%")
            row.addWidget(self.min_profit_spin)
            filter_layout.addLayout(row)
            
            # Minimum Sharpe ratio
            row = QHBoxLayout()
            row.addWidget(QLabel("Min Sharpe Ratio:"))
            self.min_sharpe_spin = QDoubleSpinBox()
            self.min_sharpe_spin.setRange(-10, 10)
            self.min_sharpe_spin.setValue(0.5)
            self.min_sharpe_spin.setSingleStep(0.1)
            row.addWidget(self.min_sharpe_spin)
            filter_layout.addLayout(row)
            
            # Minimum win rate
            row = QHBoxLayout()
            row.addWidget(QLabel("Min Win Rate %:"))
            self.min_winrate_spin = QDoubleSpinBox()
            self.min_winrate_spin.setRange(0, 100)
            self.min_winrate_spin.setValue(40)
            self.min_winrate_spin.setSuffix("%")
            row.addWidget(self.min_winrate_spin)
            filter_layout.addLayout(row)
            
            # Minimum profit factor
            row = QHBoxLayout()
            row.addWidget(QLabel("Min Profit Factor:"))
            self.min_pf_spin = QDoubleSpinBox()
            self.min_pf_spin.setRange(0, 10)
            self.min_pf_spin.setValue(1.0)
            self.min_pf_spin.setSingleStep(0.1)
            row.addWidget(self.min_pf_spin)
            filter_layout.addLayout(row)
            
            # Minimum trades
            row = QHBoxLayout()
            row.addWidget(QLabel("Min Trades:"))
            self.min_trades_spin = QSpinBox()
            self.min_trades_spin.setRange(0, 1000)
            self.min_trades_spin.setValue(10)
            row.addWidget(self.min_trades_spin)
            filter_layout.addLayout(row)
            
            # Maximum drawdown
            row = QHBoxLayout()
            row.addWidget(QLabel("Max Drawdown %:"))
            self.max_dd_spin = QDoubleSpinBox()
            self.max_dd_spin.setRange(0, 100)
            self.max_dd_spin.setValue(30)
            self.max_dd_spin.setSuffix("%")
            row.addWidget(self.max_dd_spin)
            filter_layout.addLayout(row)
            
            # Max attempts multiplier
            row = QHBoxLayout()
            row.addWidget(QLabel("Max attempts (x strategies):"))
            self.max_attempts_spin = QSpinBox()
            self.max_attempts_spin.setRange(1, 100)
            self.max_attempts_spin.setValue(10)
            self.max_attempts_spin.setToolTip("Will try up to N x requested strategies before giving up")
            row.addWidget(self.max_attempts_spin)
            filter_layout.addLayout(row)
            
            scroll_layout.addWidget(filter_group)
            
            # === Genetic Algorithm Settings ===
            genetic_group = QGroupBox("🧬 Genetic Optimization")
            genetic_layout = QVBoxLayout(genetic_group)
            
            # Population size
            row = QHBoxLayout()
            row.addWidget(QLabel("Population:"))
            self.population_spin = QSpinBox()
            self.population_spin.setRange(10, 500)
            self.population_spin.setValue(50)
            row.addWidget(self.population_spin)
            genetic_layout.addLayout(row)
            
            # Generations
            row = QHBoxLayout()
            row.addWidget(QLabel("Generations:"))
            self.generations_spin = QSpinBox()
            self.generations_spin.setRange(5, 500)
            self.generations_spin.setValue(50)
            row.addWidget(self.generations_spin)
            genetic_layout.addLayout(row)
            
            # Mutation rate
            row = QHBoxLayout()
            row.addWidget(QLabel("Mutation rate:"))
            self.mutation_spin = QDoubleSpinBox()
            self.mutation_spin.setRange(0.01, 0.5)
            self.mutation_spin.setValue(0.1)
            self.mutation_spin.setSingleStep(0.01)
            row.addWidget(self.mutation_spin)
            genetic_layout.addLayout(row)
            
            # Crossover rate
            row = QHBoxLayout()
            row.addWidget(QLabel("Crossover rate:"))
            self.crossover_spin = QDoubleSpinBox()
            self.crossover_spin.setRange(0.5, 1.0)
            self.crossover_spin.setValue(0.8)
            self.crossover_spin.setSingleStep(0.05)
            row.addWidget(self.crossover_spin)
            genetic_layout.addLayout(row)
            
            # Fitness weights
            self.use_custom_weights = QCheckBox("Custom fitness weights")
            genetic_layout.addWidget(self.use_custom_weights)
            
            # Run genetic optimization
            self.genetic_btn = QPushButton("🧬 Run Genetic Optimization")
            self.genetic_btn.clicked.connect(self.run_genetic_optimization)
            self.genetic_btn.setStyleSheet("QPushButton { padding: 10px; font-weight: bold; background-color: #1e88e5; }")
            genetic_layout.addWidget(self.genetic_btn)
            
            scroll_layout.addWidget(genetic_group)
            
            # === Validation Settings ===
            valid_group = QGroupBox("✅ Validation")
            valid_layout = QVBoxLayout(valid_group)
            
            # Walk-forward windows
            row = QHBoxLayout()
            row.addWidget(QLabel("WF Windows:"))
            self.wf_windows_spin = QSpinBox()
            self.wf_windows_spin.setRange(3, 20)
            self.wf_windows_spin.setValue(5)
            row.addWidget(self.wf_windows_spin)
            valid_layout.addLayout(row)
            
            # Train ratio
            row = QHBoxLayout()
            row.addWidget(QLabel("Train ratio:"))
            self.wf_train_ratio_spin = QDoubleSpinBox()
            self.wf_train_ratio_spin.setRange(0.5, 0.9)
            self.wf_train_ratio_spin.setValue(0.7)
            self.wf_train_ratio_spin.setSingleStep(0.05)
            row.addWidget(self.wf_train_ratio_spin)
            valid_layout.addLayout(row)
            
            # Validation buttons
            self.walkforward_btn = QPushButton("📊 Run Walk-Forward Analysis")
            self.walkforward_btn.clicked.connect(self.run_walk_forward)
            valid_layout.addWidget(self.walkforward_btn)
            
            self.overfit_btn = QPushButton("🔍 Check Overfitting")
            self.overfit_btn.clicked.connect(self.check_overfitting)
            valid_layout.addWidget(self.overfit_btn)
            
            self.montecarlo_btn = QPushButton("🎰 Monte Carlo Validation")
            self.montecarlo_btn.clicked.connect(self.run_monte_carlo)
            valid_layout.addWidget(self.montecarlo_btn)
            
            scroll_layout.addWidget(valid_group)
            
            # === Export Settings ===
            export_group = QGroupBox("📤 Export")
            export_layout = QVBoxLayout(export_group)
            
            self.export_mql5_btn = QPushButton("📄 Export to MQL5")
            self.export_mql5_btn.clicked.connect(self.export_to_mql5)
            export_layout.addWidget(self.export_mql5_btn)
            
            self.export_json_btn = QPushButton("💾 Export to JSON")
            self.export_json_btn.clicked.connect(self.export_to_json)
            export_layout.addWidget(self.export_json_btn)
            
            self.export_batch_btn = QPushButton("📁 Export All to MQL5")
            self.export_batch_btn.clicked.connect(self.export_all_to_mql5)
            export_layout.addWidget(self.export_batch_btn)
            
            scroll_layout.addWidget(export_group)
            
            # === Progress Section (Always Visible) ===
            progress_group = QGroupBox("⏳ Progress")
            progress_layout = QVBoxLayout(progress_group)
            
            # Progress bar - always visible
            self.progress_bar = QProgressBar()
            self.progress_bar.setMinimumHeight(25)
            self.progress_bar.setTextVisible(True)
            self.progress_bar.setFormat("%p% complete")
            self.progress_bar.setValue(0)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #333;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    border-radius: 3px;
                }
            """)
            progress_layout.addWidget(self.progress_bar)
            
            # Detailed status
            self.progress_detail = QLabel("")
            self.progress_detail.setStyleSheet("color: #666; font-size: 10px;")
            self.progress_detail.setWordWrap(True)
            progress_layout.addWidget(self.progress_detail)
            
            # Status label
            self.status_label = QLabel("Ready - Load data and click Generate")
            self.status_label.setStyleSheet("color: #888; font-weight: bold;")
            progress_layout.addWidget(self.status_label)
            
            # Stop button
            self.stop_btn = QPushButton("⏹️ Stop Generation")
            self.stop_btn.setEnabled(False)
            self.stop_btn.clicked.connect(self.stop_generation)
            self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
            progress_layout.addWidget(self.stop_btn)
            
            scroll_layout.addWidget(progress_group)
            
            scroll_layout.addStretch()
            scroll.setWidget(scroll_widget)
            layout.addWidget(scroll)
            
            return panel
        
        def _create_results_panel(self) -> QWidget:
            """Create the results panel."""
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # Tabs for different views
            self.results_tabs = QTabWidget()
            
            # Tab 1: Strategies list
            strategies_tab = QWidget()
            strategies_layout = QVBoxLayout(strategies_tab)
            
            self.strategies_tree = QTreeWidget()
            self.strategies_tree.setHeaderLabels([
                'Name', 'Return %', 'Sharpe', 'Win Rate', 'PF', 'Trades', 'MaxDD', 'Status'
            ])
            self.strategies_tree.setAlternatingRowColors(True)
            self.strategies_tree.itemSelectionChanged.connect(self.on_strategy_selected)
            header = self.strategies_tree.header()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            strategies_layout.addWidget(self.strategies_tree)
            
            # Action buttons for selected strategy
            btn_row = QHBoxLayout()
            self.backtest_selected_btn = QPushButton("📊 Backtest")
            self.backtest_selected_btn.clicked.connect(self.backtest_selected)
            btn_row.addWidget(self.backtest_selected_btn)
            
            self.validate_selected_btn = QPushButton("✅ Validate")
            self.validate_selected_btn.clicked.connect(self.validate_selected)
            btn_row.addWidget(self.validate_selected_btn)
            
            self.export_selected_btn = QPushButton("📤 Export")
            self.export_selected_btn.clicked.connect(self.export_selected)
            btn_row.addWidget(self.export_selected_btn)
            
            self.delete_selected_btn = QPushButton("🗑️ Delete")
            self.delete_selected_btn.clicked.connect(self.delete_selected)
            btn_row.addWidget(self.delete_selected_btn)
            
            strategies_layout.addLayout(btn_row)
            self.results_tabs.addTab(strategies_tab, "📋 Strategies")
            
            # Tab 2: Strategy details
            details_tab = QWidget()
            details_layout = QVBoxLayout(details_tab)
            
            self.details_text = QTextEdit()
            self.details_text.setReadOnly(True)
            self.details_text.setFont(QFont("Consolas", 9))
            details_layout.addWidget(self.details_text)
            
            self.results_tabs.addTab(details_tab, "📝 Details")
            
            # Tab 3: Evolution history
            evolution_tab = QWidget()
            evolution_layout = QVBoxLayout(evolution_tab)
            
            self.evolution_table = QTableWidget()
            self.evolution_table.setColumnCount(5)
            self.evolution_table.setHorizontalHeaderLabels([
                'Generation', 'Best Fitness', 'Avg Fitness', 'Best Return', 'Best Sharpe'
            ])
            self.evolution_table.setAlternatingRowColors(True)
            header = self.evolution_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            evolution_layout.addWidget(self.evolution_table)
            
            self.results_tabs.addTab(evolution_tab, "📈 Evolution")
            
            # Tab 4: Validation results
            validation_tab = QWidget()
            validation_layout = QVBoxLayout(validation_tab)
            
            self.validation_text = QTextEdit()
            self.validation_text.setReadOnly(True)
            self.validation_text.setFont(QFont("Consolas", 9))
            validation_layout.addWidget(self.validation_text)
            
            self.results_tabs.addTab(validation_tab, "✅ Validation")
            
            # Tab 5: MQL5 preview
            mql5_tab = QWidget()
            mql5_layout = QVBoxLayout(mql5_tab)
            
            self.mql5_preview = QTextEdit()
            self.mql5_preview.setReadOnly(True)
            self.mql5_preview.setFont(QFont("Consolas", 9))
            self.mql5_preview.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
            mql5_layout.addWidget(self.mql5_preview)
            
            self.results_tabs.addTab(mql5_tab, "📄 MQL5 Code")
            
            layout.addWidget(self.results_tabs)
            
            return panel
        
        def set_data(self, data):
            """Set the data to use for strategy generation."""
            self.current_data = data
            if data is not None:
                self.status_label.setText(f"Data loaded: {len(data)} candles")
        
        def generate_random_strategies(self):
            """Generate random strategies with profitability filtering."""
            if self.current_data is None or len(self.current_data) == 0:
                QMessageBox.warning(self, "No Data", "Please load data first")
                return
            
            # Reset stop flag
            self._stop_requested = False
            
            # Setup UI for generation
            self.progress_bar.setValue(0)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #333;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #2196F3;
                    border-radius: 3px;
                }
            """)
            self.generate_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            # Update generator config
            self.auto_generator.min_indicators = self.min_indicators_spin.value()
            self.auto_generator.max_indicators = self.max_indicators_spin.value()
            
            num_strategies = self.num_strategies_spin.value()
            symbol = self.symbol_combo.currentText()
            timeframe = self.timeframe_combo.currentText()
            
            # Get filter settings
            use_filters = self.enable_filters_check.isChecked()
            min_profit = self.min_profit_spin.value()
            min_sharpe = self.min_sharpe_spin.value()
            min_winrate = self.min_winrate_spin.value()
            min_pf = self.min_pf_spin.value()
            min_trades = self.min_trades_spin.value()
            max_dd = self.max_dd_spin.value()
            max_attempts = num_strategies * self.max_attempts_spin.value()
            
            accepted_count = 0
            rejected_count = 0
            attempt_count = 0
            
            # Calculate time estimates
            start_time = datetime.now()
            
            try:
                while accepted_count < num_strategies and attempt_count < max_attempts:
                    # Check for stop request
                    if self._stop_requested:
                        self.status_label.setText(f"⏹️ Stopped! Found {accepted_count} strategies")
                        break
                    
                    attempt_count += 1
                    
                    # Update progress bar
                    progress = int(accepted_count / num_strategies * 100)
                    self.progress_bar.setValue(progress)
                    self.progress_bar.setFormat(f"{progress}% - {accepted_count}/{num_strategies} found")
                    
                    # Update detailed status
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = attempt_count / elapsed if elapsed > 0 else 0
                    self.progress_detail.setText(
                        f"Attempt {attempt_count}/{max_attempts} | "
                        f"Accepted: {accepted_count} | Rejected: {rejected_count} | "
                        f"Rate: {rate:.1f}/sec"
                    )
                    self.status_label.setText(f"🔄 Generating strategy #{attempt_count}...")
                    
                    strategy = self.auto_generator.generate_random_strategy(
                        name=f"AutoGen_{datetime.now().strftime('%H%M%S')}_{attempt_count}",
                        symbols=[symbol],
                        timeframe=timeframe
                    )
                    
                    # Run backtest
                    try:
                        signals_data = self.auto_generator.evaluate_strategy(
                            strategy, self.current_data
                        )
                        result = self._quick_backtest(signals_data, symbol)
                        strategy.fitness_score = result.get('sharpe', 0)
                        strategy.backtest_results = result
                    except Exception as e:
                        logger.warning(f"Backtest failed for {strategy.name}: {e}")
                        strategy.fitness_score = -999
                        strategy.backtest_results = {'return': -100, 'sharpe': -10, 'trades': 0, 'win_rate': 0, 'profit_factor': 0, 'max_drawdown': 100}
                    
                    # Apply filters if enabled
                    if use_filters:
                        passed = self._strategy_passes_filters(
                            strategy.backtest_results,
                            min_profit, min_sharpe, min_winrate, min_pf, min_trades, max_dd
                        )
                        
                        if passed:
                            self.generated_strategies[strategy.id] = {
                                'strategy': strategy,
                                'status': 'profitable ✅'
                            }
                            accepted_count += 1
                            self.progress_bar.setStyleSheet("""
                                QProgressBar {
                                    border: 2px solid #333;
                                    border-radius: 5px;
                                    text-align: center;
                                    font-weight: bold;
                                }
                                QProgressBar::chunk {
                                    background-color: #4CAF50;
                                    border-radius: 3px;
                                }
                            """)
                        else:
                            rejected_count += 1
                    else:
                        # No filtering, accept all
                        self.generated_strategies[strategy.id] = {
                            'strategy': strategy,
                            'status': 'generated'
                        }
                        accepted_count += 1
                    
                    # Process events to keep UI responsive
                    from PyQt6.QtWidgets import QApplication
                    QApplication.processEvents()
                
                self._update_strategies_tree()
                
                # Final progress update
                self.progress_bar.setValue(100)
                elapsed = (datetime.now() - start_time).total_seconds()
                
                if self._stop_requested:
                    self.progress_bar.setFormat("Stopped")
                elif accepted_count < num_strategies:
                    self.progress_bar.setFormat(f"Completed - {accepted_count}/{num_strategies}")
                    self.progress_bar.setStyleSheet("""
                        QProgressBar {
                            border: 2px solid #333;
                            border-radius: 5px;
                            text-align: center;
                            font-weight: bold;
                        }
                        QProgressBar::chunk {
                            background-color: #FF9800;
                            border-radius: 3px;
                        }
                    """)
                    self.status_label.setText(
                        f"⚠️ Found {accepted_count}/{num_strategies} profitable strategies"
                    )
                    self.progress_detail.setText(
                        f"Completed in {elapsed:.1f}s | Attempts: {attempt_count} | Rejected: {rejected_count}"
                    )
                    QMessageBox.information(
                        self, "Generation Complete",
                        f"Found {accepted_count} profitable strategies out of {num_strategies} requested.\n"
                        f"Attempted: {attempt_count}\n"
                        f"Rejected: {rejected_count}\n"
                        f"Time: {elapsed:.1f} seconds\n\n"
                        f"Try relaxing filters or increasing max attempts."
                    )
                else:
                    self.progress_bar.setFormat("Complete!")
                    self.status_label.setText(
                        f"✅ Generated {accepted_count} profitable strategies!"
                    )
                    self.progress_detail.setText(
                        f"Completed in {elapsed:.1f}s | Attempts: {attempt_count} | Rejected: {rejected_count}"
                    )
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Generation failed: {e}")
                logger.error(f"Generation error: {e}")
                self.progress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 2px solid #333;
                        border-radius: 5px;
                        text-align: center;
                        font-weight: bold;
                    }
                    QProgressBar::chunk {
                        background-color: #f44336;
                        border-radius: 3px;
                    }
                """)
                self.status_label.setText(f"❌ Error: {str(e)[:50]}...")
            
            finally:
                self.generate_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
        
        def stop_generation(self):
            """Stop the current generation process."""
            self._stop_requested = True
            self.stop_btn.setEnabled(False)
            self.status_label.setText("⏳ Stopping...")
        
        def _strategy_passes_filters(self, results: Dict, min_profit: float, min_sharpe: float,
                                      min_winrate: float, min_pf: float, min_trades: int,
                                      max_dd: float) -> bool:
            """Check if strategy results pass the profitability filters."""
            if not results:
                return False
            
            # Check each filter
            if results.get('return', -100) < min_profit:
                return False
            
            if results.get('sharpe', -10) < min_sharpe:
                return False
            
            if results.get('win_rate', 0) < min_winrate:
                return False
            
            if results.get('profit_factor', 0) < min_pf:
                return False
            
            if results.get('trades', 0) < min_trades:
                return False
            
            if results.get('max_drawdown', 100) > max_dd:
                return False
            
            return True
        
        def _quick_backtest(self, data, symbol) -> Dict:
            """Run a quick backtest for fitness evaluation with full metrics."""
            try:
                signals = data['signal'].values if 'signal' in data.columns else []
                closes = data['close'].values
                returns = data['close'].pct_change().values
                
                if len(signals) == 0 or len(returns) == 0:
                    return self._empty_results()
                
                # Track trades and equity
                trades = []
                equity = [10000]  # Starting capital
                position = 0
                entry_price = 0
                entry_idx = 0
                
                for i in range(1, len(signals)):
                    current_signal = signals[i-1]
                    
                    # Entry
                    if position == 0:
                        if current_signal == 1:  # Buy signal
                            position = 1
                            entry_price = closes[i]
                            entry_idx = i
                        elif current_signal == -1:  # Sell signal
                            position = -1
                            entry_price = closes[i]
                            entry_idx = i
                    
                    # Exit (signal changes or opposite signal)
                    elif position != 0:
                        exit_signal = (
                            (position == 1 and current_signal == -1) or
                            (position == -1 and current_signal == 1) or
                            (position != 0 and current_signal == 0 and i > entry_idx + 5)  # Timeout exit
                        )
                        
                        if exit_signal or i == len(signals) - 1:
                            exit_price = closes[i]
                            
                            if position == 1:
                                pnl_pct = (exit_price - entry_price) / entry_price
                            else:  # Short
                                pnl_pct = (entry_price - exit_price) / entry_price
                            
                            trades.append({
                                'entry_idx': entry_idx,
                                'exit_idx': i,
                                'position': position,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'pnl_pct': pnl_pct
                            })
                            
                            position = 0
                    
                    # Update equity curve
                    if position != 0:
                        daily_return = position * returns[i]
                        equity.append(equity[-1] * (1 + daily_return))
                    else:
                        equity.append(equity[-1])
                
                if len(trades) == 0:
                    return self._empty_results()
                
                # Calculate metrics
                equity = np.array(equity)
                trade_returns = [t['pnl_pct'] for t in trades]
                
                # Total return
                total_return = (equity[-1] / equity[0] - 1) * 100
                
                # Win rate
                winning_trades = [t for t in trade_returns if t > 0]
                losing_trades = [t for t in trade_returns if t <= 0]
                win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
                
                # Profit factor
                gross_profit = sum(t for t in trade_returns if t > 0)
                gross_loss = abs(sum(t for t in trade_returns if t < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10 if gross_profit > 0 else 0)
                
                # Max drawdown
                peak = np.maximum.accumulate(equity)
                drawdown = (peak - equity) / peak * 100
                max_drawdown = np.max(drawdown)
                
                # Sharpe ratio
                daily_returns = np.diff(equity) / equity[:-1]
                if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                else:
                    sharpe = 0
                
                # Average win/loss
                avg_win = np.mean(winning_trades) * 100 if winning_trades else 0
                avg_loss = np.mean(losing_trades) * 100 if losing_trades else 0
                
                return {
                    'return': round(total_return, 2),
                    'sharpe': round(sharpe, 2),
                    'trades': len(trades),
                    'win_rate': round(win_rate, 1),
                    'profit_factor': round(profit_factor, 2),
                    'max_drawdown': round(max_drawdown, 1),
                    'avg_win': round(avg_win, 2),
                    'avg_loss': round(avg_loss, 2),
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades)
                }
                
            except Exception as e:
                logger.warning(f"Quick backtest error: {e}")
                return self._empty_results()
        
        def _empty_results(self) -> Dict:
            """Return empty results dict."""
            return {
                'return': 0,
                'sharpe': 0,
                'trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 100,
                'avg_win': 0,
                'avg_loss': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }
        
        def run_genetic_optimization(self):
            """Run genetic algorithm optimization."""
            if self.current_data is None or len(self.current_data) == 0:
                QMessageBox.warning(self, "No Data", "Please load data first")
                return
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.genetic_btn.setEnabled(False)
            self.status_label.setText("Running genetic optimization...")
            
            # Reset optimizer with new config
            self._genetic_optimizer = None
            
            symbol = self.symbol_combo.currentText()
            timeframe = self.timeframe_combo.currentText()
            
            worker = GeneticOptimizationWorker(
                optimizer=self.genetic_optimizer,
                data=self.current_data,
                backtest_engine=self.backtest_engine,
                symbols=[symbol],
                timeframe=timeframe
            )
            
            worker.signals.progress.connect(self._on_genetic_progress)
            worker.signals.result.connect(self._on_genetic_complete)
            worker.signals.error.connect(self._on_genetic_error)
            worker.signals.finished.connect(lambda: self.genetic_btn.setEnabled(True))
            
            self.thread_pool.start(worker)
        
        def _on_genetic_progress(self, progress, message):
            """Handle genetic optimization progress."""
            self.progress_bar.setValue(progress)
            self.status_label.setText(message)
        
        def _on_genetic_complete(self, result):
            """Handle genetic optimization completion."""
            self.progress_bar.setVisible(False)
            
            if result:
                # Add best strategies to list
                best_strategy = result.get('best_strategy')
                top_strategies = result.get('top_strategies', [])
                history = result.get('history', [])
                
                for strategy in top_strategies:
                    if strategy.id not in self.generated_strategies:
                        self.generated_strategies[strategy.id] = {
                            'strategy': strategy,
                            'status': 'optimized'
                        }
                
                if best_strategy:
                    self.best_strategy = best_strategy
                    self.status_label.setText(
                        f"Optimization complete! Best fitness: {best_strategy.fitness_score:.4f}"
                    )
                
                # Update evolution table
                self._update_evolution_table(history)
                self._update_strategies_tree()
                self.strategies_optimized.emit(top_strategies)
            else:
                self.status_label.setText("Optimization completed with no results")
        
        def _on_genetic_error(self, error):
            """Handle genetic optimization error."""
            self.progress_bar.setVisible(False)
            self.status_label.setText(f"Error: {error}")
            QMessageBox.critical(self, "Error", f"Optimization failed: {error}")
        
        def _update_evolution_table(self, history):
            """Update the evolution history table."""
            self.evolution_table.setRowCount(len(history))
            
            for i, gen_data in enumerate(history):
                self.evolution_table.setItem(i, 0, QTableWidgetItem(str(gen_data.get('generation', i))))
                self.evolution_table.setItem(i, 1, QTableWidgetItem(f"{gen_data.get('best_fitness', 0):.4f}"))
                self.evolution_table.setItem(i, 2, QTableWidgetItem(f"{gen_data.get('avg_fitness', 0):.4f}"))
                self.evolution_table.setItem(i, 3, QTableWidgetItem(f"{gen_data.get('best_return', 0):.2f}%"))
                self.evolution_table.setItem(i, 4, QTableWidgetItem(f"{gen_data.get('best_sharpe', 0):.2f}"))
        
        def _update_strategies_tree(self):
            """Update the strategies tree widget."""
            self.strategies_tree.clear()
            
            for strat_id, strat_data in self.generated_strategies.items():
                strategy = strat_data['strategy']
                status = strat_data['status']
                
                results = strategy.backtest_results or {}
                
                item = QTreeWidgetItem([
                    strategy.name,
                    f"{results.get('return', 0):.2f}%",
                    f"{results.get('sharpe', 0):.2f}",
                    f"{results.get('win_rate', 0):.1f}%",
                    f"{results.get('profit_factor', 0):.2f}",
                    str(results.get('trades', 0)),
                    f"{results.get('max_drawdown', 0):.1f}%",
                    status
                ])
                
                item.setData(0, Qt.ItemDataRole.UserRole, strat_id)
                
                # Color code by profitability
                ret = results.get('return', 0)
                if ret > 10:
                    item.setBackground(0, QColor(76, 175, 80, 80))  # Green
                    item.setBackground(1, QColor(76, 175, 80, 80))
                elif ret > 0:
                    item.setBackground(0, QColor(76, 175, 80, 40))  # Light green
                    item.setBackground(1, QColor(76, 175, 80, 40))
                elif ret < -10:
                    item.setBackground(0, QColor(244, 67, 54, 80))  # Red
                    item.setBackground(1, QColor(244, 67, 54, 80))
                elif ret < 0:
                    item.setBackground(0, QColor(244, 67, 54, 40))  # Light red
                    item.setBackground(1, QColor(244, 67, 54, 40))
                
                # Color status column
                if 'profitable' in status or '✅' in status:
                    item.setBackground(7, QColor(76, 175, 80, 60))
                elif 'optimized' in status:
                    item.setBackground(7, QColor(30, 136, 229, 60))
                
                self.strategies_tree.addTopLevelItem(item)
        
        def on_strategy_selected(self):
            """Handle strategy selection."""
            selected = self.strategies_tree.selectedItems()
            if not selected:
                return
            
            strat_id = selected[0].data(0, Qt.ItemDataRole.UserRole)
            if strat_id in self.generated_strategies:
                strategy = self.generated_strategies[strat_id]['strategy']
                self._show_strategy_details(strategy)
                self._show_mql5_preview(strategy)
        
        def _show_strategy_details(self, strategy):
            """Show strategy details in the details tab."""
            details = []
            details.append(f"{'='*60}")
            details.append(f"Strategy: {strategy.name}")
            details.append(f"ID: {strategy.id}")
            details.append(f"{'='*60}")
            details.append(f"\nDescription: {strategy.description}")
            details.append(f"Timeframe: {strategy.timeframe}")
            details.append(f"Symbols: {', '.join(strategy.symbols)}")
            details.append(f"Generated: {strategy.generation_date}")
            
            if strategy.backtest_results:
                results = strategy.backtest_results
                details.append(f"\n{'='*60}")
                details.append("📊 BACKTEST RESULTS")
                details.append(f"{'='*60}")
                details.append(f"\n💰 Performance:")
                details.append(f"   Net Return:     {results.get('return', 0):>10.2f}%")
                details.append(f"   Sharpe Ratio:   {results.get('sharpe', 0):>10.2f}")
                details.append(f"   Profit Factor:  {results.get('profit_factor', 0):>10.2f}")
                
                details.append(f"\n🎯 Trade Statistics:")
                details.append(f"   Total Trades:   {results.get('trades', 0):>10}")
                details.append(f"   Win Rate:       {results.get('win_rate', 0):>10.1f}%")
                details.append(f"   Winning Trades: {results.get('winning_trades', 0):>10}")
                details.append(f"   Losing Trades:  {results.get('losing_trades', 0):>10}")
                details.append(f"   Avg Win:        {results.get('avg_win', 0):>10.2f}%")
                details.append(f"   Avg Loss:       {results.get('avg_loss', 0):>10.2f}%")
                
                details.append(f"\n⚠️ Risk:")
                details.append(f"   Max Drawdown:   {results.get('max_drawdown', 0):>10.1f}%")
            
            details.append(f"\n{'='*60}")
            details.append("📊 INDICATORS")
            details.append(f"{'='*60}")
            for ind in strategy.indicators:
                details.append(f"\n• {ind.name} ({ind.indicator_type.value})")
                for param, value in ind.parameters.items():
                    details.append(f"  - {param}: {value}")
            
            details.append(f"\n{'='*60}")
            details.append("🟢 ENTRY RULES")
            details.append(f"{'='*60}")
            for rule in strategy.entry_rules:
                details.append(f"\n• {rule.name} ({rule.signal_type.name})")
                details.append(f"  Logic: {rule.logic}")
                for cond in rule.conditions:
                    details.append(f"  - {cond.left_operand} {cond.operator.value} {cond.right_operand}")
            
            details.append(f"\n{'='*60}")
            details.append("🔴 EXIT RULES")
            details.append(f"{'='*60}")
            for rule in strategy.exit_rules:
                details.append(f"\n• {rule.name} ({rule.signal_type.name})")
                details.append(f"  Logic: {rule.logic}")
                for cond in rule.conditions:
                    details.append(f"  - {cond.left_operand} {cond.operator.value} {cond.right_operand}")
            
            details.append(f"\n{'='*60}")
            details.append("⚙️ RISK MANAGEMENT")
            details.append(f"{'='*60}")
            for key, value in strategy.risk_management.items():
                details.append(f"• {key}: {value}")
            
            self.details_text.setText('\n'.join(details))
        
        def _show_mql5_preview(self, strategy):
            """Show MQL5 code preview."""
            try:
                code = self.mql5_exporter.export(strategy)
                self.mql5_preview.setText(code)
            except Exception as e:
                self.mql5_preview.setText(f"Error generating MQL5 code: {e}")
        
        def run_walk_forward(self):
            """Run walk-forward analysis on selected strategy."""
            selected = self.strategies_tree.selectedItems()
            if not selected:
                QMessageBox.warning(self, "No Selection", "Please select a strategy")
                return
            
            if self.current_data is None:
                QMessageBox.warning(self, "No Data", "Please load data first")
                return
            
            self.status_label.setText("Walk-forward analysis not yet implemented in worker")
            # TODO: Implement walk-forward worker
        
        def check_overfitting(self):
            """Check for overfitting in selected strategy."""
            selected = self.strategies_tree.selectedItems()
            if not selected:
                QMessageBox.warning(self, "No Selection", "Please select a strategy")
                return
            
            strat_id = selected[0].data(0, Qt.ItemDataRole.UserRole)
            strategy = self.generated_strategies[strat_id]['strategy']
            
            # Mock results for demonstration
            report_text = []
            report_text.append("="*50)
            report_text.append("OVERFITTING ANALYSIS REPORT")
            report_text.append("="*50)
            report_text.append(f"\nStrategy: {strategy.name}")
            report_text.append(f"\n⚠️ Note: Full analysis requires running backtest")
            report_text.append(f"\nIndicators checked:")
            report_text.append(f"• Parameter complexity: {len(strategy.indicators)} indicators")
            report_text.append(f"• Entry rules: {len(strategy.entry_rules)}")
            report_text.append(f"• Exit rules: {len(strategy.exit_rules)}")
            
            self.validation_text.setText('\n'.join(report_text))
            self.results_tabs.setCurrentIndex(3)  # Switch to validation tab
        
        def run_monte_carlo(self):
            """Run Monte Carlo validation."""
            self.status_label.setText("Monte Carlo validation not yet implemented")
        
        def backtest_selected(self):
            """Backtest the selected strategy."""
            selected = self.strategies_tree.selectedItems()
            if not selected:
                QMessageBox.warning(self, "No Selection", "Please select a strategy")
                return
            
            strat_id = selected[0].data(0, Qt.ItemDataRole.UserRole)
            strategy = self.generated_strategies[strat_id]['strategy']
            self.strategy_generated.emit(strategy.name, strategy)
        
        def validate_selected(self):
            """Validate the selected strategy."""
            self.check_overfitting()
        
        def export_selected(self):
            """Export the selected strategy."""
            self.export_to_mql5()
        
        def delete_selected(self):
            """Delete the selected strategy."""
            selected = self.strategies_tree.selectedItems()
            if not selected:
                return
            
            strat_id = selected[0].data(0, Qt.ItemDataRole.UserRole)
            
            reply = QMessageBox.question(
                self, "Confirm Delete",
                f"Delete strategy {self.generated_strategies[strat_id]['strategy'].name}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                del self.generated_strategies[strat_id]
                self._update_strategies_tree()
        
        def export_to_mql5(self):
            """Export selected strategy to MQL5."""
            selected = self.strategies_tree.selectedItems()
            if not selected:
                QMessageBox.warning(self, "No Selection", "Please select a strategy")
                return
            
            strat_id = selected[0].data(0, Qt.ItemDataRole.UserRole)
            strategy = self.generated_strategies[strat_id]['strategy']
            
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export to MQL5",
                f"{strategy.name}.mq5",
                "MQL5 Files (*.mq5);;All Files (*)"
            )
            
            if filename:
                try:
                    self.mql5_exporter.export(strategy, filename)
                    QMessageBox.information(self, "Success", f"Exported to {filename}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Export failed: {e}")
        
        def export_to_json(self):
            """Export selected strategy to JSON."""
            selected = self.strategies_tree.selectedItems()
            if not selected:
                QMessageBox.warning(self, "No Selection", "Please select a strategy")
                return
            
            strat_id = selected[0].data(0, Qt.ItemDataRole.UserRole)
            strategy = self.generated_strategies[strat_id]['strategy']
            
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export to JSON",
                f"{strategy.name}.json",
                "JSON Files (*.json);;All Files (*)"
            )
            
            if filename:
                try:
                    with open(filename, 'w') as f:
                        f.write(strategy.to_json())
                    QMessageBox.information(self, "Success", f"Exported to {filename}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Export failed: {e}")
        
        def export_all_to_mql5(self):
            """Export all strategies to MQL5."""
            if not self.generated_strategies:
                QMessageBox.warning(self, "No Strategies", "No strategies to export")
                return
            
            folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
            
            if folder:
                try:
                    strategies = [s['strategy'] for s in self.generated_strategies.values()]
                    paths = self.mql5_exporter.export_batch(strategies, folder)
                    QMessageBox.information(
                        self, "Success",
                        f"Exported {len(paths)} strategies to {folder}"
                    )
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Export failed: {e}")


# Import numpy for calculations
try:
    import numpy as np
except ImportError:
    np = None
