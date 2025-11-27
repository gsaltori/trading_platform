# ui/widgets/data_panel.py
"""
Data Panel Widget.

Manages data downloading, viewing, and preprocessing.
Supports multiple data sources including MT5 and free providers.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QLineEdit, QFormLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QDateTimeEdit,
    QSpinBox, QCheckBox, QProgressBar, QTabWidget, QTextEdit,
    QSplitter, QListWidget, QListWidgetItem, QAbstractItemView,
    QFrame, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QDateTime, QThreadPool

logger = logging.getLogger(__name__)


class DataPanel(QWidget):
    """Widget for data management and downloading."""
    
    # Signals
    data_downloaded = pyqtSignal(str, object)  # symbol, DataFrame
    data_selected = pyqtSignal(str, object)  # symbol, DataFrame
    
    # Common symbols by category
    SYMBOL_PRESETS = {
        "Forex Major": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"],
        "Forex Minor": ["EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD", "GBPAUD", "NZDJPY"],
        "Metals": ["XAUUSD", "XAGUSD"],
        "Indices": ["US30", "US500", "USTEC", "DE40", "UK100", "JP225"],
        "Crypto": ["BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD"],
        "Energy": ["XTIUSD", "XBRUSD", "XNGUSD"]
    }
    
    TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]
    
    def __init__(self, mt5_connector=None, parent=None):
        super().__init__(parent)
        self.mt5_connector = mt5_connector
        self.downloaded_data: Dict[str, pd.DataFrame] = {}
        self.thread_pool = QThreadPool()
        
        # Initialize free data manager
        self._free_data_manager = None
        
        self.setup_ui()
        self._load_cached_data_list()
    
    @property
    def free_data_manager(self):
        """Lazy load free data manager."""
        if self._free_data_manager is None:
            from data.free_data_sources import FreeDataManager
            self._free_data_manager = FreeDataManager()
        return self._free_data_manager
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create splitter for left/right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Download controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # === Data Source Selection ===
        source_group = QGroupBox("📡 Data Source")
        source_layout = QVBoxLayout(source_group)
        
        # Source dropdown
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItem("MetaTrader 5", "mt5")
        self.source_combo.addItem("Yahoo Finance (Free)", "yahoo")
        self.source_combo.addItem("Alpha Vantage (API Key)", "alphavantage")
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        source_row.addWidget(self.source_combo)
        source_layout.addLayout(source_row)
        
        # Cache options
        cache_row = QHBoxLayout()
        self.use_cache_check = QCheckBox("Use local cache")
        self.use_cache_check.setChecked(True)
        self.use_cache_check.setToolTip("Store downloaded data locally to avoid re-downloading")
        cache_row.addWidget(self.use_cache_check)
        
        self.force_download_check = QCheckBox("Force refresh")
        self.force_download_check.setToolTip("Download fresh data even if cached")
        cache_row.addWidget(self.force_download_check)
        source_layout.addLayout(cache_row)
        
        # Source status
        self.source_status = QLabel("Ready")
        self.source_status.setStyleSheet("color: #888; font-size: 10px;")
        source_layout.addWidget(self.source_status)
        
        left_layout.addWidget(source_group)
        
        # === Symbol Selection Group ===
        symbol_group = QGroupBox("📊 Symbol Selection")
        symbol_layout = QVBoxLayout(symbol_group)
        
        # Preset dropdown
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.SYMBOL_PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        
        self.add_preset_btn = QPushButton("Add All")
        self.add_preset_btn.clicked.connect(self.add_preset_symbols)
        preset_layout.addWidget(self.add_preset_btn)
        symbol_layout.addLayout(preset_layout)
        
        # Manual symbol entry
        manual_layout = QHBoxLayout()
        self.symbol_edit = QLineEdit()
        self.symbol_edit.setPlaceholderText("Enter symbol (e.g., EURUSD)")
        self.symbol_edit.returnPressed.connect(self.add_symbol)
        manual_layout.addWidget(self.symbol_edit)
        
        self.add_symbol_btn = QPushButton("Add")
        self.add_symbol_btn.clicked.connect(self.add_symbol)
        manual_layout.addWidget(self.add_symbol_btn)
        symbol_layout.addLayout(manual_layout)
        
        # Selected symbols list
        self.symbols_list = QListWidget()
        self.symbols_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.symbols_list.setMaximumHeight(100)
        symbol_layout.addWidget(self.symbols_list)
        
        # Remove button
        self.remove_symbol_btn = QPushButton("Remove Selected")
        self.remove_symbol_btn.clicked.connect(self.remove_selected_symbols)
        symbol_layout.addWidget(self.remove_symbol_btn)
        
        left_layout.addWidget(symbol_group)
        
        # === Download Parameters Group ===
        params_group = QGroupBox("⚙️ Download Parameters")
        params_layout = QFormLayout(params_group)
        
        # Timeframe
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(self.TIMEFRAMES)
        self.timeframe_combo.setCurrentText("H1")
        params_layout.addRow("Timeframe:", self.timeframe_combo)
        
        # Date range
        self.start_date = QDateTimeEdit()
        self.start_date.setDateTime(QDateTime.currentDateTime().addMonths(-6))
        self.start_date.setCalendarPopup(True)
        params_layout.addRow("Start Date:", self.start_date)
        
        self.end_date = QDateTimeEdit()
        self.end_date.setDateTime(QDateTime.currentDateTime())
        self.end_date.setCalendarPopup(True)
        params_layout.addRow("End Date:", self.end_date)
        
        # Or use candle count
        self.use_count = QCheckBox("Use candle count instead")
        self.use_count.toggled.connect(self.toggle_count_mode)
        params_layout.addRow("", self.use_count)
        
        self.candle_count = QSpinBox()
        self.candle_count.setRange(100, 100000)
        self.candle_count.setValue(5000)
        self.candle_count.setEnabled(False)
        params_layout.addRow("Candle Count:", self.candle_count)
        
        left_layout.addWidget(params_group)
        
        # Download button and progress
        self.download_btn = QPushButton("📥 Download Data")
        self.download_btn.setMinimumHeight(40)
        self.download_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        self.download_btn.clicked.connect(self.download_data)
        left_layout.addWidget(self.download_btn)
        
        # Progress section
        progress_frame = QFrame()
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v/%m")
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666;")
        progress_layout.addWidget(self.status_label)
        
        left_layout.addWidget(progress_frame)
        
        left_layout.addStretch()
        splitter.addWidget(left_widget)
        
        # Right panel - Downloaded data view
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tabs for data and cache
        self.data_tabs = QTabWidget()
        
        # Tab 1: Downloaded Data
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        
        # Data list
        data_layout.addWidget(QLabel("Downloaded Data:"))
        self.data_list = QListWidget()
        self.data_list.itemClicked.connect(self.on_data_selected)
        self.data_list.setMaximumHeight(150)
        data_layout.addWidget(self.data_list)
        
        # Data info
        self.data_info = QTextEdit()
        self.data_info.setReadOnly(True)
        self.data_info.setMaximumHeight(100)
        data_layout.addWidget(self.data_info)
        
        # Data preview table
        data_layout.addWidget(QLabel("Data Preview:"))
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels(["Time", "Open", "High", "Low", "Close", "Volume"])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        data_layout.addWidget(self.data_table)
        
        # Data operations
        ops_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("📤 Export CSV")
        self.export_btn.clicked.connect(self.export_data)
        self.export_btn.setEnabled(False)
        ops_layout.addWidget(self.export_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear All")
        self.clear_btn.clicked.connect(self.clear_all_data)
        ops_layout.addWidget(self.clear_btn)
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.refresh_selected_data)
        self.refresh_btn.setEnabled(False)
        ops_layout.addWidget(self.refresh_btn)
        
        data_layout.addLayout(ops_layout)
        self.data_tabs.addTab(data_tab, "📋 Data")
        
        # Tab 2: Cache Management
        cache_tab = QWidget()
        cache_layout = QVBoxLayout(cache_tab)
        
        # Cache stats
        self.cache_stats_label = QLabel("Cache: 0 entries, 0 MB")
        cache_layout.addWidget(self.cache_stats_label)
        
        # Cached data list
        cache_layout.addWidget(QLabel("Cached Data:"))
        self.cache_list = QListWidget()
        self.cache_list.setMaximumHeight(200)
        cache_layout.addWidget(self.cache_list)
        
        # Cache operations
        cache_ops = QHBoxLayout()
        
        self.load_cached_btn = QPushButton("📂 Load Selected")
        self.load_cached_btn.clicked.connect(self.load_from_cache)
        cache_ops.addWidget(self.load_cached_btn)
        
        self.refresh_cache_btn = QPushButton("🔄 Refresh List")
        self.refresh_cache_btn.clicked.connect(self._load_cached_data_list)
        cache_ops.addWidget(self.refresh_cache_btn)
        
        self.clear_cache_btn = QPushButton("🗑️ Clear Cache")
        self.clear_cache_btn.clicked.connect(self.clear_cache)
        cache_ops.addWidget(self.clear_cache_btn)
        
        cache_layout.addLayout(cache_ops)
        cache_layout.addStretch()
        
        self.data_tabs.addTab(cache_tab, "💾 Cache")
        
        right_layout.addWidget(self.data_tabs)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([350, 450])
        
        layout.addWidget(splitter)
    
    def on_source_changed(self, index):
        """Handle data source change."""
        source = self.source_combo.currentData()
        
        if source == "mt5":
            if self.mt5_connector and self.mt5_connector.connected:
                self.source_status.setText("✅ Connected to MT5")
                self.source_status.setStyleSheet("color: green; font-size: 10px;")
            else:
                self.source_status.setText("⚠️ MT5 not connected")
                self.source_status.setStyleSheet("color: orange; font-size: 10px;")
        elif source == "yahoo":
            try:
                import yfinance
                self.source_status.setText("✅ Yahoo Finance available")
                self.source_status.setStyleSheet("color: green; font-size: 10px;")
            except ImportError:
                self.source_status.setText("❌ Install: pip install yfinance")
                self.source_status.setStyleSheet("color: red; font-size: 10px;")
        elif source == "alphavantage":
            self.source_status.setText("🔑 Requires free API key from alphavantage.co")
            self.source_status.setStyleSheet("color: blue; font-size: 10px;")
    
    def on_preset_changed(self, preset_name: str):
        """Handle preset selection change."""
        pass  # Just update UI, don't add automatically
    
    def add_preset_symbols(self):
        """Add all symbols from current preset."""
        preset_name = self.preset_combo.currentText()
        symbols = self.SYMBOL_PRESETS.get(preset_name, [])
        
        for symbol in symbols:
            self.add_symbol_to_list(symbol)
    
    def add_symbol(self):
        """Add symbol from text input."""
        symbol = self.symbol_edit.text().strip().upper()
        if symbol:
            self.add_symbol_to_list(symbol)
            self.symbol_edit.clear()
    
    def add_symbol_to_list(self, symbol: str):
        """Add symbol to the list if not already present."""
        # Check if already in list
        for i in range(self.symbols_list.count()):
            if self.symbols_list.item(i).text() == symbol:
                return
        
        self.symbols_list.addItem(symbol)
    
    def remove_selected_symbols(self):
        """Remove selected symbols from list."""
        for item in self.symbols_list.selectedItems():
            self.symbols_list.takeItem(self.symbols_list.row(item))
    
    def toggle_count_mode(self, checked: bool):
        """Toggle between date range and candle count mode."""
        self.candle_count.setEnabled(checked)
        self.start_date.setEnabled(not checked)
        self.end_date.setEnabled(not checked)
    
    def download_data(self):
        """Download data for all selected symbols."""
        symbols = [self.symbols_list.item(i).text() 
                  for i in range(self.symbols_list.count())]
        
        if not symbols:
            QMessageBox.warning(self, "No Symbols", "Please add at least one symbol")
            return
        
        source = self.source_combo.currentData()
        timeframe = self.timeframe_combo.currentText()
        
        # Setup progress
        self.download_btn.setEnabled(False)
        self.progress_bar.setMaximum(len(symbols))
        self.progress_bar.setValue(0)
        
        # Get date parameters
        if self.use_count.isChecked():
            count = self.candle_count.value()
            start_date = datetime.now() - timedelta(days=365*2)  # 2 years back for count mode
            end_date = datetime.now()
        else:
            count = None
            start_date = self.start_date.dateTime().toPyDateTime()
            end_date = self.end_date.dateTime().toPyDateTime()
        
        force_download = self.force_download_check.isChecked()
        use_cache = self.use_cache_check.isChecked()
        
        # Download based on source
        if source == "mt5":
            self._download_from_mt5(symbols, timeframe, start_date, end_date, count)
        else:
            self._download_from_free_source(
                symbols, timeframe, start_date, end_date, 
                source, force_download, use_cache
            )
    
    def _download_from_mt5(self, symbols, timeframe, start_date, end_date, count):
        """Download from MetaTrader 5."""
        if not self.mt5_connector or not self.mt5_connector.connected:
            QMessageBox.warning(self, "Not Connected", "Please connect to MT5 first")
            self.download_btn.setEnabled(True)
            return
        
        # Download in thread
        from ..utils.workers import MultiSymbolDataWorker
        
        worker = MultiSymbolDataWorker(
            self.mt5_connector,
            symbols,
            timeframe,
            start_date,
            end_date,
            count
        )
        
        worker.signals.progress.connect(self.on_download_progress)
        worker.signals.result.connect(self.on_download_complete)
        worker.signals.error.connect(self.on_download_error)
        worker.signals.log.connect(self.on_worker_log)
        
        self.thread_pool.start(worker)
    
    def _download_from_free_source(self, symbols, timeframe, start_date, end_date,
                                    source, force_download, use_cache):
        """Download from free data source."""
        self.free_data_manager.use_cache = use_cache
        
        results = {}
        total = len(symbols)
        
        for i, symbol in enumerate(symbols):
            self.progress_bar.setValue(i)
            self.status_label.setText(f"Downloading {symbol}...")
            QApplication.processEvents()
            
            try:
                def progress_cb(pct, msg):
                    self.status_label.setText(f"{symbol}: {msg}")
                    QApplication.processEvents()
                
                data = self.free_data_manager.get_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    source=source,
                    force_download=force_download,
                    progress_callback=progress_cb
                )
                
                if data is not None and not data.empty:
                    results[symbol] = data
                    self.status_label.setText(f"✅ {symbol}: {len(data)} candles")
                else:
                    self.status_label.setText(f"⚠️ {symbol}: No data")
                    
            except Exception as e:
                self.status_label.setText(f"❌ {symbol}: {str(e)}")
                logger.error(f"Error downloading {symbol}: {e}")
            
            QApplication.processEvents()
        
        self.progress_bar.setValue(total)
        self.on_download_complete(results)
        self._load_cached_data_list()
    
    def on_download_progress(self, percentage: int, message: str):
        """Handle download progress."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
    
    def on_download_complete(self, results: Dict[str, pd.DataFrame]):
        """Handle download completion."""
        self.download_btn.setEnabled(True)
        
        count = 0
        for symbol, data in results.items():
            if data is not None and not data.empty:
                self.downloaded_data[symbol] = data
                self.data_downloaded.emit(symbol, data)
                count += 1
        
        self.update_data_list()
        self.status_label.setText(f"✅ Downloaded data for {count} symbol(s)")
    
    def on_download_error(self, error: str, traceback: str):
        """Handle download error."""
        self.download_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error}")
        QMessageBox.critical(self, "Download Error", f"Error downloading data:\n{error}")
    
    def on_worker_log(self, level: str, message: str):
        """Handle worker log messages."""
        logger.log(getattr(logging, level, logging.INFO), message)
    
    def update_data_list(self):
        """Update the downloaded data list."""
        self.data_list.clear()
        
        for symbol, data in self.downloaded_data.items():
            item = QListWidgetItem(f"{symbol} ({len(data):,} candles)")
            item.setData(Qt.ItemDataRole.UserRole, symbol)
            self.data_list.addItem(item)
    
    def on_data_selected(self, item: QListWidgetItem):
        """Handle data selection."""
        symbol = item.data(Qt.ItemDataRole.UserRole)
        data = self.downloaded_data.get(symbol)
        
        if data is not None:
            self.show_data_preview(symbol, data)
            self.export_btn.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            self.data_selected.emit(symbol, data)
    
    def show_data_preview(self, symbol: str, data: pd.DataFrame):
        """Show data preview in table and info."""
        # Update info
        info_text = f"""Symbol: {symbol}
Timeframe: {self.timeframe_combo.currentText()}
Records: {len(data):,}
Date Range: {data.index[0]} to {data.index[-1]}
Columns: {', '.join(data.columns)}

Statistics:
  Open  - Min: {data['open'].min():.5f}, Max: {data['open'].max():.5f}
  High  - Min: {data['high'].min():.5f}, Max: {data['high'].max():.5f}
  Low   - Min: {data['low'].min():.5f}, Max: {data['low'].max():.5f}
  Close - Min: {data['close'].min():.5f}, Max: {data['close'].max():.5f}"""
        
        self.data_info.setText(info_text)
        
        # Update table (show last 100 rows)
        preview_data = data.tail(100)
        self.data_table.setRowCount(len(preview_data))
        
        for i, (idx, row) in enumerate(preview_data.iterrows()):
            self.data_table.setItem(i, 0, QTableWidgetItem(str(idx)))
            self.data_table.setItem(i, 1, QTableWidgetItem(f"{row['open']:.5f}"))
            self.data_table.setItem(i, 2, QTableWidgetItem(f"{row['high']:.5f}"))
            self.data_table.setItem(i, 3, QTableWidgetItem(f"{row['low']:.5f}"))
            self.data_table.setItem(i, 4, QTableWidgetItem(f"{row['close']:.5f}"))
            self.data_table.setItem(i, 5, QTableWidgetItem(str(row.get('volume', 0))))
    
    def _load_cached_data_list(self):
        """Load list of cached data."""
        self.cache_list.clear()
        
        try:
            cached = self.free_data_manager.get_cached_symbols()
            stats = self.free_data_manager.get_cache_stats()
            
            self.cache_stats_label.setText(
                f"Cache: {stats['entries']} entries, {stats['total_size_mb']} MB"
            )
            
            for entry in cached:
                text = f"{entry['symbol']} {entry['timeframe']} ({entry['source']}) - {entry['records']:,} candles"
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, entry)
                self.cache_list.addItem(item)
                
        except Exception as e:
            logger.warning(f"Error loading cache list: {e}")
    
    def load_from_cache(self):
        """Load selected cached data."""
        current_item = self.cache_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a cached entry")
            return
        
        entry = current_item.data(Qt.ItemDataRole.UserRole)
        symbol = entry['symbol']
        timeframe = entry['timeframe']
        source = entry['source']
        
        # Get from cache
        data = self.free_data_manager.cache.get(symbol, timeframe, source)
        
        if data is not None:
            self.downloaded_data[symbol] = data
            self.update_data_list()
            self.data_downloaded.emit(symbol, data)
            self.status_label.setText(f"Loaded {symbol} from cache ({len(data):,} candles)")
            self.data_tabs.setCurrentIndex(0)  # Switch to data tab
        else:
            QMessageBox.warning(self, "Error", "Could not load data from cache")
    
    def clear_cache(self):
        """Clear all cached data."""
        reply = QMessageBox.question(
            self, "Confirm Clear",
            "Clear all cached data? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.free_data_manager.clear_cache()
            self._load_cached_data_list()
            self.status_label.setText("Cache cleared")
    
    def export_data(self):
        """Export selected data to CSV."""
        current_item = self.data_list.currentItem()
        if not current_item:
            return
        
        symbol = current_item.data(Qt.ItemDataRole.UserRole)
        data = self.downloaded_data.get(symbol)
        
        if data is None:
            return
        
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Data",
            f"{symbol}_{self.timeframe_combo.currentText()}.csv",
            "CSV Files (*.csv)"
        )
        
        if filename:
            data.to_csv(filename)
            QMessageBox.information(self, "Export Complete", f"Data exported to {filename}")
    
    def clear_all_data(self):
        """Clear all downloaded data."""
        reply = QMessageBox.question(
            self, "Confirm Clear",
            "Clear all downloaded data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.downloaded_data.clear()
            self.data_list.clear()
            self.data_table.setRowCount(0)
            self.data_info.clear()
            self.export_btn.setEnabled(False)
            self.refresh_btn.setEnabled(False)
    
    def refresh_selected_data(self):
        """Refresh data for selected symbol."""
        current_item = self.data_list.currentItem()
        if not current_item:
            return
        
        symbol = current_item.data(Qt.ItemDataRole.UserRole)
        
        # Add to download list and download
        self.symbols_list.clear()
        self.add_symbol_to_list(symbol)
        self.force_download_check.setChecked(True)  # Force refresh
        self.download_data()
        self.force_download_check.setChecked(False)
    
    def get_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get downloaded data for a symbol."""
        return self.downloaded_data.get(symbol)
    
    def get_all_data(self) -> Dict[str, pd.DataFrame]:
        """Get all downloaded data."""
        return self.downloaded_data.copy()
    
    def set_mt5_connector(self, connector):
        """Set the MT5 connector."""
        self.mt5_connector = connector
        self.on_source_changed(0)  # Update status
