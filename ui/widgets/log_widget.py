# ui/widgets/log_widget.py
"""
Log Widget.

Displays application logs with filtering and search capabilities.
"""

import logging
from datetime import datetime
from typing import List, Dict
from collections import deque

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QComboBox, QLineEdit, QLabel, QCheckBox, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QTextCharFormat, QColor, QFont, QTextCursor


class QTextEditLogger(logging.Handler, QObject):
    """Custom logging handler that emits to a signal."""
    
    log_signal = pyqtSignal(str, str, str)  # level, message, timestamp
    
    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        
        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        ))
    
    def emit(self, record):
        msg = self.format(record)
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_signal.emit(record.levelname, msg, timestamp)


class LogWidget(QWidget):
    """Widget for displaying and filtering logs."""
    
    # Log level colors
    LEVEL_COLORS = {
        'DEBUG': QColor('#6c757d'),
        'INFO': QColor('#0d6efd'),
        'WARNING': QColor('#ffc107'),
        'ERROR': QColor('#dc3545'),
        'CRITICAL': QColor('#dc3545')
    }
    
    def __init__(self, max_lines: int = 1000, parent=None):
        super().__init__(parent)
        self.max_lines = max_lines
        self.log_buffer: deque = deque(maxlen=max_lines)
        self.current_filter_level = 'DEBUG'
        self.search_text = ''
        
        self.setup_ui()
        self.setup_logging()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Controls
        controls = QHBoxLayout()
        
        # Log level filter
        controls.addWidget(QLabel("Level:"))
        self.level_combo = QComboBox()
        self.level_combo.addItems(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
        self.level_combo.setCurrentText('DEBUG')
        self.level_combo.currentTextChanged.connect(self.on_level_changed)
        controls.addWidget(self.level_combo)
        
        # Search
        controls.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Filter logs...")
        self.search_edit.textChanged.connect(self.on_search_changed)
        self.search_edit.setMaximumWidth(200)
        controls.addWidget(self.search_edit)
        
        # Auto-scroll
        self.auto_scroll_check = QCheckBox("Auto-scroll")
        self.auto_scroll_check.setChecked(True)
        controls.addWidget(self.auto_scroll_check)
        
        controls.addStretch()
        
        # Clear button
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self.clear_logs)
        controls.addWidget(self.clear_btn)
        
        # Export button
        self.export_btn = QPushButton("📤 Export")
        self.export_btn.clicked.connect(self.export_logs)
        controls.addWidget(self.export_btn)
        
        layout.addLayout(controls)
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 9))
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
            }
        """)
        layout.addWidget(self.log_display)
        
        # Status bar
        status = QHBoxLayout()
        self.line_count_label = QLabel("Lines: 0")
        status.addWidget(self.line_count_label)
        status.addStretch()
        layout.addLayout(status)
    
    def setup_logging(self):
        """Setup logging handler."""
        self.log_handler = QTextEditLogger()
        self.log_handler.log_signal.connect(self.append_log)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        
        # Also capture platform-specific loggers
        for logger_name in ['data', 'strategies', 'backtesting', 'ml', 'execution']:
            logger = logging.getLogger(logger_name)
            logger.addHandler(self.log_handler)
    
    def append_log(self, level: str, message: str, timestamp: str):
        """Append a log message to the display."""
        # Store in buffer
        self.log_buffer.append({
            'level': level,
            'message': message,
            'timestamp': timestamp
        })
        
        # Check if should display based on filter
        if not self.should_display(level, message):
            return
        
        # Get color for level
        color = self.LEVEL_COLORS.get(level, QColor('#d4d4d4'))
        
        # Format and append
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Create format
        fmt = QTextCharFormat()
        fmt.setForeground(color)
        
        cursor.insertText(message + '\n', fmt)
        
        # Auto-scroll
        if self.auto_scroll_check.isChecked():
            self.log_display.setTextCursor(cursor)
            self.log_display.ensureCursorVisible()
        
        # Update line count
        self.line_count_label.setText(f"Lines: {len(self.log_buffer)}")
    
    def should_display(self, level: str, message: str) -> bool:
        """Check if message should be displayed based on filters."""
        # Check level
        level_order = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if level_order.index(level) < level_order.index(self.current_filter_level):
            return False
        
        # Check search text
        if self.search_text and self.search_text.lower() not in message.lower():
            return False
        
        return True
    
    def on_level_changed(self, level: str):
        """Handle log level filter change."""
        self.current_filter_level = level
        self.refresh_display()
    
    def on_search_changed(self, text: str):
        """Handle search text change."""
        self.search_text = text
        self.refresh_display()
    
    def refresh_display(self):
        """Refresh the log display with current filters."""
        self.log_display.clear()
        
        for log_entry in self.log_buffer:
            if self.should_display(log_entry['level'], log_entry['message']):
                color = self.LEVEL_COLORS.get(log_entry['level'], QColor('#d4d4d4'))
                
                cursor = self.log_display.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                
                fmt = QTextCharFormat()
                fmt.setForeground(color)
                
                cursor.insertText(log_entry['message'] + '\n', fmt)
        
        # Scroll to end
        if self.auto_scroll_check.isChecked():
            self.log_display.moveCursor(QTextCursor.MoveOperation.End)
    
    def clear_logs(self):
        """Clear all logs."""
        self.log_buffer.clear()
        self.log_display.clear()
        self.line_count_label.setText("Lines: 0")
    
    def export_logs(self):
        """Export logs to file."""
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Logs",
            f"trading_platform_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)"
        )
        
        if filename:
            with open(filename, 'w') as f:
                for log_entry in self.log_buffer:
                    f.write(log_entry['message'] + '\n')
    
    def log(self, level: str, message: str):
        """Manually log a message."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted = f"{timestamp} - GUI - {level} - {message}"
        self.append_log(level, formatted, timestamp)
    
    def info(self, message: str):
        """Log info message."""
        self.log('INFO', message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.log('WARNING', message)
    
    def error(self, message: str):
        """Log error message."""
        self.log('ERROR', message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.log('DEBUG', message)
