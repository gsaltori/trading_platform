# ui/main_window.py
"""
Main Window for the Trading Platform GUI.

Integrates all widgets into a comprehensive trading interface.
"""

import sys
import logging
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# Check if PyQt6 is available
PYQT_AVAILABLE = False
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QDockWidget, QToolBar, QStatusBar, QMenuBar,
        QMenu, QMessageBox, QSplitter, QLabel, QProgressBar,
        QFileDialog, QInputDialog
    )
    from PyQt6.QtCore import Qt, QTimer, QSettings, QSize
    from PyQt6.QtGui import QAction, QIcon, QFont, QKeySequence
    PYQT_AVAILABLE = True
except ImportError:
    logger.warning("PyQt6 not available, GUI disabled")


def run_gui():
    """Run the GUI application."""
    if not PYQT_AVAILABLE:
        print("=" * 60)
        print("GUI NOT AVAILABLE")
        print("=" * 60)
        print("PyQt6 is not installed. To use the GUI, install it with:")
        print("  pip install PyQt6")
        print("")
        print("For charts, also install:")
        print("  pip install matplotlib")
        print("")
        print("For now, you can use the platform in headless mode:")
        print("  python main.py --headless")
        print("=" * 60)
        return 1
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("Trading Platform")
    app.setOrganizationName("TradingPlatform")
    
    # Set application-wide font
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    # Apply dark theme
    app.setStyleSheet(get_dark_theme())
    
    window = MainWindow()
    window.show()
    
    return app.exec()


def get_dark_theme() -> str:
    """Get dark theme stylesheet."""
    return """
    QMainWindow {
        background-color: #2b2b2b;
    }
    QWidget {
        background-color: #2b2b2b;
        color: #e0e0e0;
    }
    QGroupBox {
        border: 1px solid #3c3c3c;
        border-radius: 5px;
        margin-top: 10px;
        padding-top: 10px;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }
    QPushButton {
        background-color: #3c3c3c;
        border: 1px solid #5c5c5c;
        border-radius: 4px;
        padding: 6px 12px;
        min-width: 60px;
    }
    QPushButton:hover {
        background-color: #4c4c4c;
        border-color: #6c6c6c;
    }
    QPushButton:pressed {
        background-color: #2c2c2c;
    }
    QPushButton:disabled {
        background-color: #2c2c2c;
        color: #6c6c6c;
    }
    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background-color: #1e1e1e;
        border: 1px solid #3c3c3c;
        border-radius: 4px;
        padding: 4px;
        selection-background-color: #264f78;
    }
    QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border-color: #0078d4;
    }
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    QComboBox::down-arrow {
        width: 12px;
        height: 12px;
    }
    QTableWidget {
        background-color: #1e1e1e;
        alternate-background-color: #252525;
        gridline-color: #3c3c3c;
        border: 1px solid #3c3c3c;
    }
    QTableWidget::item:selected {
        background-color: #264f78;
    }
    QHeaderView::section {
        background-color: #2b2b2b;
        border: 1px solid #3c3c3c;
        padding: 4px;
    }
    QTabWidget::pane {
        border: 1px solid #3c3c3c;
        border-radius: 4px;
    }
    QTabBar::tab {
        background-color: #2b2b2b;
        border: 1px solid #3c3c3c;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        padding: 8px 16px;
        margin-right: 2px;
    }
    QTabBar::tab:selected {
        background-color: #3c3c3c;
    }
    QTabBar::tab:hover {
        background-color: #353535;
    }
    QProgressBar {
        border: 1px solid #3c3c3c;
        border-radius: 4px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #0078d4;
        border-radius: 3px;
    }
    QScrollBar:vertical {
        background-color: #2b2b2b;
        width: 12px;
        margin: 0;
    }
    QScrollBar::handle:vertical {
        background-color: #5c5c5c;
        border-radius: 6px;
        min-height: 20px;
    }
    QScrollBar::handle:vertical:hover {
        background-color: #6c6c6c;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
    }
    QDockWidget {
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }
    QDockWidget::title {
        background-color: #353535;
        padding: 6px;
    }
    QMenuBar {
        background-color: #2b2b2b;
        border-bottom: 1px solid #3c3c3c;
    }
    QMenuBar::item:selected {
        background-color: #3c3c3c;
    }
    QMenu {
        background-color: #2b2b2b;
        border: 1px solid #3c3c3c;
    }
    QMenu::item:selected {
        background-color: #3c3c3c;
    }
    QToolBar {
        background-color: #2b2b2b;
        border: none;
        spacing: 3px;
        padding: 3px;
    }
    QStatusBar {
        background-color: #007acc;
        color: white;
    }
    QListWidget {
        background-color: #1e1e1e;
        border: 1px solid #3c3c3c;
        border-radius: 4px;
    }
    QListWidget::item:selected {
        background-color: #264f78;
    }
    QListWidget::item:hover {
        background-color: #353535;
    }
    QTreeWidget {
        background-color: #1e1e1e;
        border: 1px solid #3c3c3c;
        border-radius: 4px;
    }
    QTreeWidget::item:selected {
        background-color: #264f78;
    }
    QSplitter::handle {
        background-color: #3c3c3c;
    }
    QSplitter::handle:horizontal {
        width: 2px;
    }
    QSplitter::handle:vertical {
        height: 2px;
    }
    """


if PYQT_AVAILABLE:
    from .widgets.mt5_connection_widget import MT5ConnectionWidget
    from .widgets.data_panel import DataPanel
    from .widgets.strategy_generator_widget import StrategyGeneratorWidget
    from .widgets.backtest_results_widget import BacktestResultsWidget
    from .widgets.charts_widget import ChartsWidget
    from .widgets.log_widget import LogWidget
    from .widgets.auto_generator_widget import AutoGeneratorWidget
    
    class MainWindow(QMainWindow):
        """Main window for the trading platform."""
        
        def __init__(self):
            super().__init__()
            
            # Initialize components
            self.mt5_connector = None
            self.strategy_engine = None
            self.ml_engine = None
            self.backtest_engine = None
            self.config = None
            
            # Initialize engines
            self.init_engines()
            
            # Setup UI
            self.init_ui()
            self.create_menus()
            self.create_toolbars()
            self.create_status_bar()
            self.setup_connections()
            
            # Load settings
            self.load_settings()
            
            # Log startup
            self.log_widget.info("Trading Platform started")
        
        def init_engines(self):
            """Initialize trading engines."""
            try:
                # Config
                from config.settings import ConfigManager
                self.config = ConfigManager()
                
                # MT5 Connector
                from data.mt5_connector import MT5ConnectionManager
                self.mt5_connector = MT5ConnectionManager(self.config)
                
                # Strategy Engine
                from strategies.strategy_engine import StrategyEngine
                self.strategy_engine = StrategyEngine()
                
                # ML Engine
                from ml.ml_engine import MLEngine
                self.ml_engine = MLEngine()
                
                # Backtest Engine
                from backtesting.backtest_engine import BacktestEngine
                self.backtest_engine = BacktestEngine(
                    initial_capital=10000.0
                )
                
                logger.info("All engines initialized successfully")
                
            except Exception as e:
                logger.error(f"Error initializing engines: {e}")
        
        def init_ui(self):
            """Initialize the user interface."""
            self.setWindowTitle("Trading Platform - Strategy Generator")
            self.setMinimumSize(1400, 900)
            
            # Central widget with tabs
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            main_layout = QVBoxLayout(central_widget)
            main_layout.setContentsMargins(5, 5, 5, 5)
            
            # Main splitter
            main_splitter = QSplitter(Qt.Orientation.Horizontal)
            
            # Left panel - MT5 Connection
            left_panel = QWidget()
            left_layout = QVBoxLayout(left_panel)
            left_layout.setContentsMargins(0, 0, 0, 0)
            
            self.mt5_widget = MT5ConnectionWidget(self.mt5_connector)
            left_layout.addWidget(self.mt5_widget)
            
            left_panel.setMaximumWidth(400)
            main_splitter.addWidget(left_panel)
            
            # Center panel - Main tabs
            center_panel = QWidget()
            center_layout = QVBoxLayout(center_panel)
            center_layout.setContentsMargins(0, 0, 0, 0)
            
            self.main_tabs = QTabWidget()
            
            # Data Tab
            self.data_panel = DataPanel(self.mt5_connector)
            self.main_tabs.addTab(self.data_panel, "📥 Data")
            
            # Strategy Generator Tab
            self.strategy_widget = StrategyGeneratorWidget(
                self.strategy_engine, self.ml_engine
            )
            self.main_tabs.addTab(self.strategy_widget, "🔧 Strategy Generator")
            
            # Backtest Tab
            self.backtest_widget = BacktestResultsWidget(self.backtest_engine)
            self.main_tabs.addTab(self.backtest_widget, "📊 Backtest")
            
            # Charts Tab
            self.charts_widget = ChartsWidget()
            self.main_tabs.addTab(self.charts_widget, "📈 Charts")
            
            # Auto Generator Tab (NEW)
            self.auto_generator_widget = AutoGeneratorWidget()
            self.main_tabs.addTab(self.auto_generator_widget, "🧬 Auto Generator")
            
            center_layout.addWidget(self.main_tabs)
            main_splitter.addWidget(center_panel)
            
            # Set splitter sizes
            main_splitter.setSizes([350, 1050])
            
            main_layout.addWidget(main_splitter)
            
            # Bottom panel - Logs (as dock widget)
            self.log_dock = QDockWidget("Logs", self)
            self.log_widget = LogWidget()
            self.log_dock.setWidget(self.log_widget)
            self.log_dock.setAllowedAreas(
                Qt.DockWidgetArea.BottomDockWidgetArea | 
                Qt.DockWidgetArea.TopDockWidgetArea
            )
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
            
            # Set initial dock size
            self.log_dock.setMinimumHeight(150)
            self.log_dock.setMaximumHeight(300)
        
        def create_menus(self):
            """Create menu bar."""
            menubar = self.menuBar()
            
            # File Menu
            file_menu = menubar.addMenu("&File")
            
            new_action = QAction("&New Project", self)
            new_action.setShortcut(QKeySequence.StandardKey.New)
            new_action.triggered.connect(self.new_project)
            file_menu.addAction(new_action)
            
            open_action = QAction("&Open Project", self)
            open_action.setShortcut(QKeySequence.StandardKey.Open)
            open_action.triggered.connect(self.open_project)
            file_menu.addAction(open_action)
            
            save_action = QAction("&Save Project", self)
            save_action.setShortcut(QKeySequence.StandardKey.Save)
            save_action.triggered.connect(self.save_project)
            file_menu.addAction(save_action)
            
            file_menu.addSeparator()
            
            import_data_action = QAction("Import &Data", self)
            import_data_action.triggered.connect(self.import_data)
            file_menu.addAction(import_data_action)
            
            export_data_action = QAction("&Export Data", self)
            export_data_action.triggered.connect(self.export_data)
            file_menu.addAction(export_data_action)
            
            file_menu.addSeparator()
            
            exit_action = QAction("E&xit", self)
            exit_action.setShortcut(QKeySequence.StandardKey.Quit)
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)
            
            # View Menu
            view_menu = menubar.addMenu("&View")
            
            toggle_logs_action = QAction("Toggle &Logs", self)
            toggle_logs_action.setShortcut("Ctrl+L")
            toggle_logs_action.triggered.connect(self.toggle_logs)
            view_menu.addAction(toggle_logs_action)
            
            view_menu.addSeparator()
            
            # Theme submenu
            theme_menu = view_menu.addMenu("&Theme")
            
            dark_theme_action = QAction("&Dark", self)
            dark_theme_action.triggered.connect(lambda: self.set_theme('dark'))
            theme_menu.addAction(dark_theme_action)
            
            light_theme_action = QAction("&Light", self)
            light_theme_action.triggered.connect(lambda: self.set_theme('light'))
            theme_menu.addAction(light_theme_action)
            
            # Tools Menu
            tools_menu = menubar.addMenu("&Tools")
            
            scan_mt5_action = QAction("&Scan MT5 Installations", self)
            scan_mt5_action.triggered.connect(self.mt5_widget.scan_installations)
            tools_menu.addAction(scan_mt5_action)
            
            tools_menu.addSeparator()
            
            settings_action = QAction("&Settings", self)
            settings_action.triggered.connect(self.show_settings)
            tools_menu.addAction(settings_action)
            
            # Help Menu
            help_menu = menubar.addMenu("&Help")
            
            docs_action = QAction("&Documentation", self)
            docs_action.triggered.connect(self.show_documentation)
            help_menu.addAction(docs_action)
            
            help_menu.addSeparator()
            
            about_action = QAction("&About", self)
            about_action.triggered.connect(self.show_about)
            help_menu.addAction(about_action)
        
        def create_toolbars(self):
            """Create tool bars."""
            # Main toolbar
            main_toolbar = QToolBar("Main")
            main_toolbar.setMovable(False)
            main_toolbar.setIconSize(QSize(24, 24))
            self.addToolBar(main_toolbar)
            
            # Quick connect button
            self.quick_connect_btn = QAction("🔌 Connect MT5", self)
            self.quick_connect_btn.setToolTip("Quick connect to MT5")
            self.quick_connect_btn.triggered.connect(self.quick_connect)
            main_toolbar.addAction(self.quick_connect_btn)
            
            main_toolbar.addSeparator()
            
            # Download data button
            download_btn = QAction("📥 Download Data", self)
            download_btn.setToolTip("Download market data")
            download_btn.triggered.connect(lambda: self.main_tabs.setCurrentWidget(self.data_panel))
            main_toolbar.addAction(download_btn)
            
            # Generate strategy button
            generate_btn = QAction("🔧 Generate Strategy", self)
            generate_btn.setToolTip("Generate trading strategy")
            generate_btn.triggered.connect(lambda: self.main_tabs.setCurrentWidget(self.strategy_widget))
            main_toolbar.addAction(generate_btn)
            
            # Run backtest button
            backtest_btn = QAction("📊 Backtest", self)
            backtest_btn.setToolTip("Run backtest")
            backtest_btn.triggered.connect(self.run_quick_backtest)
            main_toolbar.addAction(backtest_btn)
            
            main_toolbar.addSeparator()
            
            # View charts button
            charts_btn = QAction("📈 Charts", self)
            charts_btn.setToolTip("View charts")
            charts_btn.triggered.connect(lambda: self.main_tabs.setCurrentWidget(self.charts_widget))
            main_toolbar.addAction(charts_btn)
        
        def create_status_bar(self):
            """Create status bar."""
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            
            # MT5 status
            self.mt5_status_label = QLabel("MT5: Disconnected")
            self.status_bar.addWidget(self.mt5_status_label)
            
            # Separator
            self.status_bar.addWidget(QLabel(" | "))
            
            # Data status
            self.data_status_label = QLabel("Data: No data loaded")
            self.status_bar.addWidget(self.data_status_label)
            
            # Permanent widgets on right
            self.status_bar.addPermanentWidget(QLabel("v1.0.0"))
        
        def setup_connections(self):
            """Setup signal/slot connections."""
            # MT5 connections
            self.mt5_widget.connection_changed.connect(self.on_mt5_connection_changed)
            self.mt5_widget.symbols_loaded.connect(self.on_symbols_loaded)
            
            # Data connections
            self.data_panel.data_downloaded.connect(self.on_data_downloaded)
            self.data_panel.data_selected.connect(self.on_data_selected)
            
            # Strategy connections
            self.strategy_widget.strategy_generated.connect(self.on_strategy_generated)
            
            # Backtest connections
            self.backtest_widget.backtest_requested.connect(self.on_backtest_requested)
            
            # Auto Generator connections
            self.auto_generator_widget.strategy_generated.connect(self.on_auto_strategy_generated)
            self.auto_generator_widget.strategies_optimized.connect(self.on_strategies_optimized)
        
        def on_mt5_connection_changed(self, connected: bool):
            """Handle MT5 connection change."""
            if connected:
                self.mt5_status_label.setText("MT5: Connected ✓")
                self.mt5_status_label.setStyleSheet("color: #4caf50;")
                self.quick_connect_btn.setText("🔌 Disconnect MT5")
                self.log_widget.info("Connected to MetaTrader 5")
            else:
                self.mt5_status_label.setText("MT5: Disconnected")
                self.mt5_status_label.setStyleSheet("color: #f44336;")
                self.quick_connect_btn.setText("🔌 Connect MT5")
                self.log_widget.info("Disconnected from MetaTrader 5")
        
        def on_symbols_loaded(self, symbols: list):
            """Handle symbols loaded from MT5."""
            self.backtest_widget.set_symbols(symbols)
            self.log_widget.info(f"Loaded {len(symbols)} symbols from MT5")
        
        def on_data_downloaded(self, symbol: str, data):
            """Handle data download completion."""
            self.data_status_label.setText(f"Data: {symbol} ({len(data)} candles)")
            self.log_widget.info(f"Downloaded {len(data)} candles for {symbol}")
            
            # Update strategy generator with data
            self.strategy_widget.set_data(data)
            
            # Update auto generator with data
            self.auto_generator_widget.set_data(data)
        
        def on_data_selected(self, symbol: str, data):
            """Handle data selection."""
            # Plot on charts
            self.charts_widget.plot_price_data(data)
            
            # Update strategy generator
            self.strategy_widget.set_data(data)
            
            # Update auto generator
            self.auto_generator_widget.set_data(data)
        
        def on_strategy_generated(self, name: str, strategy):
            """Handle strategy generation."""
            self.log_widget.info(f"Strategy '{name}' generated")
            
            # If we have data, offer to backtest
            all_data = self.data_panel.get_all_data()
            if all_data:
                reply = QMessageBox.question(
                    self, "Run Backtest?",
                    f"Strategy '{name}' generated. Run backtest?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Get first available data
                    symbol = list(all_data.keys())[0]
                    data = all_data[symbol]
                    
                    self.main_tabs.setCurrentWidget(self.backtest_widget)
                    self.backtest_widget.run_backtest(strategy, data, symbol)
        
        def on_auto_strategy_generated(self, name: str, strategy):
            """Handle auto-generated strategy."""
            self.log_widget.info(f"Auto-generated strategy '{name}' ready")
            
            # Offer to backtest
            all_data = self.data_panel.get_all_data()
            if all_data:
                reply = QMessageBox.question(
                    self, "Run Backtest?",
                    f"Strategy '{name}' generated. Run backtest?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    symbol = list(all_data.keys())[0]
                    data = all_data[symbol]
                    self.main_tabs.setCurrentWidget(self.backtest_widget)
                    # Note: Would need to adapt strategy for backtest
        
        def on_strategies_optimized(self, strategies: list):
            """Handle optimized strategies from genetic algorithm."""
            self.log_widget.info(f"Received {len(strategies)} optimized strategies")
        
        def on_backtest_requested(self, strategy, data):
            """Handle backtest request."""
            # Get selected strategy from strategy widget
            current_strategy = None
            for name, strat_data in self.strategy_widget.generated_strategies.items():
                if 'strategy' in strat_data:
                    current_strategy = strat_data['strategy']
                    break
            
            if current_strategy is None:
                QMessageBox.warning(
                    self, "No Strategy",
                    "Please generate a strategy first"
                )
                return
            
            # Get data
            all_data = self.data_panel.get_all_data()
            if not all_data:
                QMessageBox.warning(
                    self, "No Data",
                    "Please download data first"
                )
                return
            
            symbol = list(all_data.keys())[0]
            data = all_data[symbol]
            
            self.backtest_widget.run_backtest(current_strategy, data, symbol)
        
        def quick_connect(self):
            """Quick connect/disconnect to MT5."""
            if self.mt5_widget.is_mt5_connected():
                self.mt5_widget.disconnect_mt5()
            else:
                self.mt5_widget.connect_mt5()
        
        def run_quick_backtest(self):
            """Run a quick backtest with current strategy and data."""
            self.main_tabs.setCurrentWidget(self.backtest_widget)
            self.backtest_widget.request_backtest()
        
        def toggle_logs(self):
            """Toggle log panel visibility."""
            if self.log_dock.isVisible():
                self.log_dock.hide()
            else:
                self.log_dock.show()
        
        def set_theme(self, theme: str):
            """Set application theme."""
            if theme == 'dark':
                QApplication.instance().setStyleSheet(get_dark_theme())
            else:
                QApplication.instance().setStyleSheet("")
        
        def new_project(self):
            """Create a new project."""
            reply = QMessageBox.question(
                self, "New Project",
                "Create a new project? Unsaved changes will be lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.data_panel.clear_all_data()
                self.strategy_widget.generated_strategies.clear()
                self.strategy_widget.update_strategies_tree()
                self.charts_widget.clear_charts()
                self.log_widget.info("New project created")
        
        def open_project(self):
            """Open an existing project."""
            filename, _ = QFileDialog.getOpenFileName(
                self, "Open Project",
                "",
                "Trading Project (*.tproj);;All Files (*)"
            )
            
            if filename:
                self.log_widget.info(f"Opening project: {filename}")
                # TODO: Implement project loading
        
        def save_project(self):
            """Save current project."""
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Project",
                "project.tproj",
                "Trading Project (*.tproj);;All Files (*)"
            )
            
            if filename:
                self.log_widget.info(f"Saving project: {filename}")
                # TODO: Implement project saving
        
        def import_data(self):
            """Import data from file."""
            filename, _ = QFileDialog.getOpenFileName(
                self, "Import Data",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if filename:
                try:
                    import pandas as pd
                    data = pd.read_csv(filename, parse_dates=['time'], index_col='time')
                    
                    # Get symbol name from filename
                    symbol = Path(filename).stem.upper()
                    
                    self.data_panel.downloaded_data[symbol] = data
                    self.data_panel.update_data_list()
                    
                    self.log_widget.info(f"Imported {len(data)} rows from {filename}")
                    
                except Exception as e:
                    QMessageBox.critical(self, "Import Error", f"Error importing data: {e}")
        
        def export_data(self):
            """Export current data."""
            self.data_panel.export_data()
        
        def show_settings(self):
            """Show settings dialog."""
            QMessageBox.information(
                self, "Settings",
                "Settings dialog coming soon!"
            )
        
        def show_documentation(self):
            """Show documentation."""
            QMessageBox.information(
                self, "Documentation",
                "Documentation:\n\n"
                "1. Connect to MT5 using the left panel\n"
                "2. Download data for your desired symbols\n"
                "3. Generate strategies using rule-based or ML methods\n"
                "4. Backtest your strategies\n"
                "5. Analyze results and optimize\n\n"
                "For more information, visit the project repository."
            )
        
        def show_about(self):
            """Show about dialog."""
            QMessageBox.about(
                self,
                "About Trading Platform",
                "<h2>Trading Platform</h2>"
                "<p>Version 1.0.0</p>"
                "<p>A comprehensive algorithmic trading platform with:</p>"
                "<ul>"
                "<li>MetaTrader 5 integration</li>"
                "<li>Automatic strategy generation</li>"
                "<li>Machine Learning optimization</li>"
                "<li>Advanced backtesting</li>"
                "</ul>"
                "<p>© 2024 Trading Platform</p>"
            )
        
        def load_settings(self):
            """Load application settings."""
            settings = QSettings()
            
            # Window geometry
            geometry = settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
            
            # Window state
            state = settings.value("windowState")
            if state:
                self.restoreState(state)
        
        def save_settings(self):
            """Save application settings."""
            settings = QSettings()
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("windowState", self.saveState())
        
        def closeEvent(self, event):
            """Handle window close event."""
            # Disconnect MT5
            if self.mt5_widget.is_mt5_connected():
                self.mt5_widget.disconnect_mt5()
            
            # Save settings
            self.save_settings()
            
            self.log_widget.info("Trading Platform closed")
            event.accept()


else:
    class MainWindow:
        """Dummy MainWindow when PyQt6 is not available."""
        def __init__(self):
            raise ImportError("PyQt6 is not installed")


if __name__ == "__main__":
    run_gui()
