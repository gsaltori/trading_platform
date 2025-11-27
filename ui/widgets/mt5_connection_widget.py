# ui/widgets/mt5_connection_widget.py
"""
MetaTrader 5 Connection Widget.

Manages MT5 installations, connections, and account information.
"""

import logging
from typing import Optional, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QLineEdit, QFormLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QFileDialog,
    QDialog, QDialogButtonBox, QSpinBox, QCheckBox, QFrame,
    QSplitter, QTextEdit, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QIcon

from ..utils.mt5_discovery import MT5Discovery, MT5Installation

logger = logging.getLogger(__name__)


class AddInstallationDialog(QDialog):
    """Dialog for adding a new MT5 installation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add MT5 Installation")
        self.setMinimumWidth(450)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        form = QFormLayout()
        
        # Path selection
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Path to MT5 installation folder")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_path)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_btn)
        form.addRow("Installation Path:", path_layout)
        
        # Name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Display name (optional)")
        form.addRow("Name:", self.name_edit)
        
        # Server
        self.server_edit = QLineEdit()
        self.server_edit.setPlaceholderText("Trading server")
        form.addRow("Server:", self.server_edit)
        
        # Login
        self.login_spin = QSpinBox()
        self.login_spin.setRange(0, 999999999)
        self.login_spin.setSpecialValueText("Not set")
        form.addRow("Login:", self.login_spin)
        
        # Password
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("Password (not stored)")
        form.addRow("Password:", self.password_edit)
        
        # Portable mode
        self.portable_check = QCheckBox("Portable installation")
        form.addRow("", self.portable_check)
        
        layout.addLayout(form)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def browse_path(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select MT5 Installation Directory"
        )
        if path:
            self.path_edit.setText(path)
    
    def get_data(self) -> dict:
        return {
            'path': self.path_edit.text(),
            'name': self.name_edit.text(),
            'server': self.server_edit.text(),
            'login': self.login_spin.value(),
            'password': self.password_edit.text(),
            'portable': self.portable_check.isChecked()
        }


class MT5ConnectionWidget(QWidget):
    """Widget for managing MT5 connections."""
    
    # Signals
    connection_changed = pyqtSignal(bool)  # Connected/disconnected
    installation_selected = pyqtSignal(object)  # MT5Installation
    account_updated = pyqtSignal(dict)  # Account info
    symbols_loaded = pyqtSignal(list)  # List of symbols
    
    def __init__(self, mt5_connector=None, parent=None):
        super().__init__(parent)
        self.mt5_connector = mt5_connector
        self.discovery = MT5Discovery()
        self.current_installation: Optional[MT5Installation] = None
        self.is_connected = False
        
        self.setup_ui()
        self.load_installations()
        
        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_account_info)
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Installations Group
        install_group = QGroupBox("MT5 Installations")
        install_layout = QVBoxLayout(install_group)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        self.scan_btn = QPushButton("🔍 Scan")
        self.scan_btn.setToolTip("Scan for MT5 installations")
        self.scan_btn.clicked.connect(self.scan_installations)
        
        self.add_btn = QPushButton("➕ Add")
        self.add_btn.setToolTip("Add installation manually")
        self.add_btn.clicked.connect(self.add_installation)
        
        self.remove_btn = QPushButton("🗑️ Remove")
        self.remove_btn.setToolTip("Remove selected installation")
        self.remove_btn.clicked.connect(self.remove_installation)
        self.remove_btn.setEnabled(False)
        
        toolbar.addWidget(self.scan_btn)
        toolbar.addWidget(self.add_btn)
        toolbar.addWidget(self.remove_btn)
        toolbar.addStretch()
        
        install_layout.addLayout(toolbar)
        
        # Installations Table
        self.installations_table = QTableWidget()
        self.installations_table.setColumnCount(4)
        self.installations_table.setHorizontalHeaderLabels(["Name", "Path", "Server", "Status"])
        self.installations_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.installations_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.installations_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.installations_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.installations_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.installations_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.installations_table.itemSelectionChanged.connect(self.on_installation_selected)
        self.installations_table.setMaximumHeight(150)
        
        install_layout.addWidget(self.installations_table)
        layout.addWidget(install_group)
        
        # Connection Group
        conn_group = QGroupBox("Connection")
        conn_layout = QVBoxLayout(conn_group)
        
        # Credentials Form
        cred_form = QFormLayout()
        
        self.server_combo = QComboBox()
        self.server_combo.setEditable(True)
        self.server_combo.setPlaceholderText("Select or enter server")
        cred_form.addRow("Server:", self.server_combo)
        
        self.login_edit = QLineEdit()
        self.login_edit.setPlaceholderText("Account login")
        cred_form.addRow("Login:", self.login_edit)
        
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("Account password")
        cred_form.addRow("Password:", self.password_edit)
        
        conn_layout.addLayout(cred_form)
        
        # Connection buttons
        conn_buttons = QHBoxLayout()
        
        self.connect_btn = QPushButton("🔌 Connect")
        self.connect_btn.clicked.connect(self.toggle_connection)
        self.connect_btn.setMinimumHeight(35)
        
        self.status_label = QLabel("⚪ Disconnected")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        conn_buttons.addWidget(self.connect_btn)
        conn_buttons.addWidget(self.status_label)
        
        conn_layout.addLayout(conn_buttons)
        layout.addWidget(conn_group)
        
        # Account Info Group
        account_group = QGroupBox("Account Information")
        account_layout = QFormLayout(account_group)
        
        self.account_login_label = QLabel("-")
        account_layout.addRow("Login:", self.account_login_label)
        
        self.account_balance_label = QLabel("-")
        self.account_balance_label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        account_layout.addRow("Balance:", self.account_balance_label)
        
        self.account_equity_label = QLabel("-")
        self.account_equity_label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        account_layout.addRow("Equity:", self.account_equity_label)
        
        self.account_margin_label = QLabel("-")
        account_layout.addRow("Free Margin:", self.account_margin_label)
        
        self.account_profit_label = QLabel("-")
        account_layout.addRow("Open P&L:", self.account_profit_label)
        
        self.account_leverage_label = QLabel("-")
        account_layout.addRow("Leverage:", self.account_leverage_label)
        
        self.account_server_label = QLabel("-")
        account_layout.addRow("Server:", self.account_server_label)
        
        layout.addWidget(account_group)
        
        # Symbols Quick View
        symbols_group = QGroupBox("Available Symbols")
        symbols_layout = QVBoxLayout(symbols_group)
        
        self.symbols_search = QLineEdit()
        self.symbols_search.setPlaceholderText("Search symbols...")
        self.symbols_search.textChanged.connect(self.filter_symbols)
        symbols_layout.addWidget(self.symbols_search)
        
        self.symbols_table = QTableWidget()
        self.symbols_table.setColumnCount(3)
        self.symbols_table.setHorizontalHeaderLabels(["Symbol", "Spread", "Contract Size"])
        self.symbols_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.symbols_table.setMaximumHeight(150)
        symbols_layout.addWidget(self.symbols_table)
        
        layout.addWidget(symbols_group)
        layout.addStretch()
    
    def load_installations(self):
        """Load saved installations into the table."""
        self.installations_table.setRowCount(0)
        
        for inst in self.discovery.installations:
            self.add_installation_to_table(inst)
    
    def add_installation_to_table(self, installation: MT5Installation):
        """Add an installation to the table."""
        row = self.installations_table.rowCount()
        self.installations_table.insertRow(row)
        
        # Name
        name_item = QTableWidgetItem(installation.name)
        name_item.setData(Qt.ItemDataRole.UserRole, installation)
        self.installations_table.setItem(row, 0, name_item)
        
        # Path
        self.installations_table.setItem(row, 1, QTableWidgetItem(installation.path))
        
        # Server
        self.installations_table.setItem(row, 2, QTableWidgetItem(installation.server or "-"))
        
        # Status
        status = "✅ Valid" if installation.is_valid else "❌ Invalid"
        status_item = QTableWidgetItem(status)
        if not installation.is_valid:
            status_item.setForeground(QColor("red"))
        self.installations_table.setItem(row, 3, status_item)
    
    def scan_installations(self):
        """Scan for MT5 installations."""
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning...")
        
        try:
            installations = self.discovery.discover_installations()
            self.load_installations()
            
            QMessageBox.information(
                self, "Scan Complete",
                f"Found {len(installations)} MT5 installation(s)"
            )
        except Exception as e:
            QMessageBox.warning(self, "Scan Error", f"Error scanning: {e}")
        finally:
            self.scan_btn.setEnabled(True)
            self.scan_btn.setText("🔍 Scan")
    
    def add_installation(self):
        """Show dialog to add installation manually."""
        dialog = AddInstallationDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            
            if not data['path']:
                QMessageBox.warning(self, "Error", "Please specify installation path")
                return
            
            installation = self.discovery.add_installation(
                path=data['path'],
                name=data['name'],
                server=data['server'],
                login=data['login'],
                password=data['password']
            )
            
            if installation:
                self.load_installations()
                QMessageBox.information(self, "Success", "Installation added successfully")
            else:
                QMessageBox.warning(self, "Error", "Could not add installation. Check the path.")
    
    def remove_installation(self):
        """Remove selected installation."""
        current_row = self.installations_table.currentRow()
        if current_row < 0:
            return
        
        item = self.installations_table.item(current_row, 0)
        installation = item.data(Qt.ItemDataRole.UserRole)
        
        reply = QMessageBox.question(
            self, "Confirm Removal",
            f"Remove {installation.name} from the list?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.discovery.remove_installation(installation.terminal_exe)
            self.load_installations()
    
    def on_installation_selected(self):
        """Handle installation selection."""
        current_row = self.installations_table.currentRow()
        self.remove_btn.setEnabled(current_row >= 0)
        
        if current_row >= 0:
            item = self.installations_table.item(current_row, 0)
            installation = item.data(Qt.ItemDataRole.UserRole)
            self.current_installation = installation
            
            # Update connection form
            if installation.server:
                self.server_combo.setCurrentText(installation.server)
            if installation.login:
                self.login_edit.setText(str(installation.login))
            
            self.installation_selected.emit(installation)
    
    def toggle_connection(self):
        """Connect or disconnect from MT5."""
        if self.is_connected:
            self.disconnect_mt5()
        else:
            self.connect_mt5()
    
    def connect_mt5(self):
        """Connect to MT5."""
        if not self.current_installation:
            QMessageBox.warning(self, "Error", "Please select an installation first")
            return
        
        if not self.mt5_connector:
            QMessageBox.warning(self, "Error", "MT5 connector not initialized")
            return
        
        self.connect_btn.setEnabled(False)
        self.status_label.setText("🟡 Connecting...")
        
        try:
            # Update connector config
            if hasattr(self.mt5_connector, 'config'):
                self.mt5_connector.config.mt5.path = self.current_installation.terminal_exe
                self.mt5_connector.config.mt5.server = self.server_combo.currentText()
                
                login_text = self.login_edit.text()
                if login_text:
                    self.mt5_connector.config.mt5.login = int(login_text)
                
                self.mt5_connector.config.mt5.password = self.password_edit.text()
            
            # Connect
            if self.mt5_connector.initialize():
                self.is_connected = True
                self.connect_btn.setText("🔌 Disconnect")
                self.status_label.setText("🟢 Connected")
                
                # Get account info
                self.refresh_account_info()
                
                # Load symbols
                self.load_symbols()
                
                # Start refresh timer
                self.refresh_timer.start(5000)  # Refresh every 5 seconds
                
                self.connection_changed.emit(True)
            else:
                self.status_label.setText("🔴 Connection Failed")
                QMessageBox.warning(self, "Connection Failed", "Could not connect to MT5")
                
        except Exception as e:
            self.status_label.setText("🔴 Error")
            QMessageBox.critical(self, "Error", f"Connection error: {e}")
        finally:
            self.connect_btn.setEnabled(True)
    
    def disconnect_mt5(self):
        """Disconnect from MT5."""
        self.refresh_timer.stop()
        
        if self.mt5_connector:
            self.mt5_connector.shutdown()
        
        self.is_connected = False
        self.connect_btn.setText("🔌 Connect")
        self.status_label.setText("⚪ Disconnected")
        
        # Clear account info
        self.clear_account_info()
        
        self.connection_changed.emit(False)
    
    def refresh_account_info(self):
        """Refresh account information."""
        if not self.is_connected or not self.mt5_connector:
            return
        
        try:
            account_info = self.mt5_connector.get_account_info()
            
            if account_info:
                currency = account_info.get('currency', 'USD')
                
                self.account_login_label.setText(str(account_info.get('login', '-')))
                self.account_balance_label.setText(f"{account_info.get('balance', 0):,.2f} {currency}")
                self.account_equity_label.setText(f"{account_info.get('equity', 0):,.2f} {currency}")
                self.account_margin_label.setText(f"{account_info.get('free_margin', 0):,.2f} {currency}")
                
                profit = account_info.get('profit', 0)
                profit_text = f"{profit:+,.2f} {currency}"
                self.account_profit_label.setText(profit_text)
                if profit > 0:
                    self.account_profit_label.setStyleSheet("color: green;")
                elif profit < 0:
                    self.account_profit_label.setStyleSheet("color: red;")
                else:
                    self.account_profit_label.setStyleSheet("")
                
                self.account_leverage_label.setText(f"1:{account_info.get('leverage', '-')}")
                self.account_server_label.setText(account_info.get('server', '-'))
                
                self.account_updated.emit(account_info)
                
        except Exception as e:
            logger.warning(f"Error refreshing account info: {e}")
    
    def clear_account_info(self):
        """Clear account information labels."""
        self.account_login_label.setText("-")
        self.account_balance_label.setText("-")
        self.account_equity_label.setText("-")
        self.account_margin_label.setText("-")
        self.account_profit_label.setText("-")
        self.account_profit_label.setStyleSheet("")
        self.account_leverage_label.setText("-")
        self.account_server_label.setText("-")
    
    def load_symbols(self):
        """Load available symbols."""
        if not self.is_connected or not self.mt5_connector:
            return
        
        try:
            symbols = list(self.mt5_connector.symbols_cache.keys())
            
            self.symbols_table.setRowCount(0)
            
            for symbol_name in sorted(symbols)[:100]:  # Limit to 100 for performance
                symbol_info = self.mt5_connector.symbols_cache.get(symbol_name)
                if symbol_info:
                    row = self.symbols_table.rowCount()
                    self.symbols_table.insertRow(row)
                    
                    self.symbols_table.setItem(row, 0, QTableWidgetItem(symbol_info.name))
                    self.symbols_table.setItem(row, 1, QTableWidgetItem(str(symbol_info.spread)))
                    self.symbols_table.setItem(row, 2, QTableWidgetItem(str(symbol_info.trade_contract_size)))
            
            self.symbols_loaded.emit(symbols)
            
        except Exception as e:
            logger.warning(f"Error loading symbols: {e}")
    
    def filter_symbols(self, text: str):
        """Filter symbols table by search text."""
        for row in range(self.symbols_table.rowCount()):
            item = self.symbols_table.item(row, 0)
            if item:
                match = text.lower() in item.text().lower()
                self.symbols_table.setRowHidden(row, not match)
    
    def get_selected_installation(self) -> Optional[MT5Installation]:
        """Get the currently selected installation."""
        return self.current_installation
    
    def is_mt5_connected(self) -> bool:
        """Check if MT5 is connected."""
        return self.is_connected
