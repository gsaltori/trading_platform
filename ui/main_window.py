# ui/main_window.py
import sys
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
                             QTextEdit, QProgressBar, QSplitter, QGroupBox,
                             QGridLayout, QMessageBox, QToolBar, QStatusBar,
                             QFileDialog, QCheckBox, QSpinBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QFont
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QDateTimeAxis, QValueAxis
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from core.platform import get_platform

logger = logging.getLogger(__name__)

class DataWorker(QThread):
    """Hilo para cargar datos sin bloquear la interfaz"""
    data_loaded = pyqtSignal(pd.DataFrame)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, symbol, timeframe, days):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.days = days
    
    def run(self):
        try:
            platform = get_platform()
            data = platform.get_market_data(self.symbol, self.timeframe, self.days)
            if data is not None:
                self.data_loaded.emit(data)
            else:
                self.error_occurred.emit("No se pudieron cargar los datos")
        except Exception as e:
            self.error_occurred.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.platform = get_platform()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Plataforma de Trading Algorítmico")
        self.setGeometry(100, 100, 1400, 900)
        
        # Crear barra de herramientas
        self.create_toolbar()
        
        # Crear barra de estado
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo")
        
        # Crear widget central con pestañas
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Crear las diferentes pestañas
        self.create_dashboard_tab()
        self.create_data_tab()
        self.create_strategy_editor_tab()
        self.create_backtesting_tab()
        self.create_live_trading_tab()
        
        # Timer para actualizaciones en tiempo real
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_realtime_data)
        self.timer.start(5000)  # Actualizar cada 5 segundos
        
    def create_toolbar(self):
        toolbar = QToolBar("Barra principal")
        self.addToolBar(toolbar)
        
        # Acciones
        connect_action = QAction("Conectar MT5", self)
        connect_action.triggered.connect(self.connect_mt5)
        toolbar.addAction(connect_action)
        
        toolbar.addSeparator()
        
        refresh_action = QAction("Actualizar datos", self)
        refresh_action.triggered.connect(self.refresh_data)
        toolbar.addAction(refresh_action)
        
    def create_dashboard_tab(self):
        """Pestaña del Dashboard principal"""
        dashboard_tab = QWidget()
        layout = QVBoxLayout()
        
        # Título
        title = QLabel("Dashboard Principal")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Métricas en tiempo real
        metrics_group = QGroupBox("Métricas de Cuenta")
        metrics_layout = QGridLayout()
        
        self.balance_label = QLabel("Balance: --")
        self.equity_label = QLabel("Equity: --")
        self.margin_label = QLabel("Margen: --")
        self.free_margin_label = QLabel("Margen Libre: --")
        
        metrics_layout.addWidget(QLabel("Balance:"), 0, 0)
        metrics_layout.addWidget(self.balance_label, 0, 1)
        metrics_layout.addWidget(QLabel("Equity:"), 1, 0)
        metrics_layout.addWidget(self.equity_label, 1, 1)
        metrics_layout.addWidget(QLabel("Margen:"), 2, 0)
        metrics_layout.addWidget(self.margin_label, 2, 1)
        metrics_layout.addWidget(QLabel("Margen Libre:"), 3, 0)
        metrics_layout.addWidget(self.free_margin_label, 3, 1)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Gráfico de equity
        self.equity_chart = QChart()
        self.equity_series = QLineSeries()
        self.equity_chart.addSeries(self.equity_series)
        
        axis_x = QDateTimeAxis()
        axis_x.setFormat("dd/MM hh:mm")
        self.equity_chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        self.equity_series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        self.equity_chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        self.equity_series.attachAxis(axis_y)
        
        self.equity_chart_view = QChartView(self.equity_chart)
        layout.addWidget(self.equity_chart_view)
        
        dashboard_tab.setLayout(layout)
        self.tabs.addTab(dashboard_tab, "Dashboard")
        
    def create_data_tab(self):
        """Pestaña de gestión de datos"""
        data_tab = QWidget()
        layout = QVBoxLayout()
        
        # Controles de datos
        controls_layout = QHBoxLayout()
        
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "XAUUSD"])
        controls_layout.addWidget(QLabel("Símbolo:"))
        controls_layout.addWidget(self.symbol_combo)
        
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
        controls_layout.addWidget(QLabel("Timeframe:"))
        controls_layout.addWidget(self.timeframe_combo)
        
        self.days_spin = QSpinBox()
        self.days_spin.setRange(1, 365)
        self.days_spin.setValue(30)
        controls_layout.addWidget(QLabel("Días:"))
        controls_layout.addWidget(self.days_spin)
        
        load_btn = QPushButton("Cargar Datos")
        load_btn.clicked.connect(self.load_data)
        controls_layout.addWidget(load_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Tabla de datos
        self.data_table = QTableWidget()
        layout.addWidget(self.data_table)
        
        data_tab.setLayout(layout)
        self.tabs.addTab(data_tab, "Datos de Mercado")
        
    def create_strategy_editor_tab(self):
        """Pestaña del editor de estrategias"""
        editor_tab = QWidget()
        layout = QVBoxLayout()
        
        # Aquí irá el editor visual de estrategias (para la Fase 2, un editor simple)
        self.strategy_code_editor = QTextEdit()
        self.strategy_code_editor.setPlaceholderText("Escribe tu estrategia en Python aquí...")
        layout.addWidget(self.strategy_code_editor)
        
        # Botones para guardar y cargar estrategias
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Guardar Estrategia")
        save_btn.clicked.connect(self.save_strategy)
        btn_layout.addWidget(save_btn)
        
        load_btn = QPushButton("Cargar Estrategia")
        load_btn.clicked.connect(self.load_strategy)
        btn_layout.addWidget(load_btn)
        
        test_btn = QPushButton("Probar Estrategia")
        test_btn.clicked.connect(self.test_strategy)
        btn_layout.addWidget(test_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        editor_tab.setLayout(layout)
        self.tabs.addTab(editor_tab, "Editor de Estrategias")
        
    def create_backtesting_tab(self):
        """Pestaña de backtesting"""
        backtesting_tab = QWidget()
        layout = QVBoxLayout()
        
        # Controles de backtesting
        controls_layout = QGridLayout()
        
        self.bt_symbol_combo = QComboBox()
        self.bt_symbol_combo.addItems(["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "XAUUSD"])
        controls_layout.addWidget(QLabel("Símbolo:"), 0, 0)
        controls_layout.addWidget(self.bt_symbol_combo, 0, 1)
        
        self.bt_timeframe_combo = QComboBox()
        self.bt_timeframe_combo.addItems(["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
        controls_layout.addWidget(QLabel("Timeframe:"), 0, 2)
        controls_layout.addWidget(self.bt_timeframe_combo, 0, 3)
        
        self.bt_start_date = QLineEdit()
        self.bt_start_date.setPlaceholderText("YYYY-MM-DD")
        controls_layout.addWidget(QLabel("Fecha Inicio:"), 1, 0)
        controls_layout.addWidget(self.bt_start_date, 1, 1)
        
        self.bt_end_date = QLineEdit()
        self.bt_end_date.setPlaceholderText("YYYY-MM-DD")
        controls_layout.addWidget(QLabel("Fecha Fin:"), 1, 2)
        controls_layout.addWidget(self.bt_end_date, 1, 3)
        
        self.bt_initial_capital = QDoubleSpinBox()
        self.bt_initial_capital.setRange(100, 1000000)
        self.bt_initial_capital.setValue(10000)
        controls_layout.addWidget(QLabel("Capital Inicial:"), 2, 0)
        controls_layout.addWidget(self.bt_initial_capital, 2, 1)
        
        run_bt_btn = QPushButton("Ejecutar Backtest")
        run_bt_btn.clicked.connect(self.run_backtest)
        controls_layout.addWidget(run_bt_btn, 2, 2, 1, 2)
        
        layout.addLayout(controls_layout)
        
        # Resultados del backtest
        self.bt_results_text = QTextEdit()
        self.bt_results_text.setReadOnly(True)
        layout.addWidget(self.bt_results_text)
        
        backtesting_tab.setLayout(layout)
        self.tabs.addTab(backtesting_tab, "Backtesting")
        
    def create_live_trading_tab(self):
        """Pestaña de trading en vivo"""
        live_tab = QWidget()
        layout = QVBoxLayout()
        
        # Controles de trading en vivo
        controls_layout = QHBoxLayout()
        
        self.live_strategy_combo = QComboBox()
        # Aquí cargaremos las estrategias disponibles
        controls_layout.addWidget(QLabel("Estrategia:"))
        controls_layout.addWidget(self.live_strategy_combo)
        
        self.start_live_btn = QPushButton("Iniciar Trading")
        self.start_live_btn.clicked.connect(self.toggle_live_trading)
        controls_layout.addWidget(self.start_live_btn)
        
        self.live_status_label = QLabel("Detenido")
        self.live_status_label.setStyleSheet("color: red; font-weight: bold;")
        controls_layout.addWidget(self.live_status_label)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Log de operaciones en vivo
        self.live_log = QTextEdit()
        self.live_log.setReadOnly(True)
        layout.addWidget(self.live_log)
        
        live_tab.setLayout(layout)
        self.tabs.addTab(live_tab, "Trading en Vivo")
        
    def connect_mt5(self):
        """Conectar/Desconectar MT5"""
        try:
            if not self.platform.initialized:
                if self.platform.initialize():
                    self.status_bar.showMessage("MT5 Conectado")
                    QMessageBox.information(self, "Conexión", "Conectado a MT5 correctamente")
                else:
                    QMessageBox.critical(self, "Error", "No se pudo conectar a MT5")
            else:
                self.platform.shutdown()
                self.status_bar.showMessage("MT5 Desconectado")
                QMessageBox.information(self, "Conexión", "Desconectado de MT5")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error de conexión: {str(e)}")
    
    def load_data(self):
        """Cargar datos en segundo plano"""
        symbol = self.symbol_combo.currentText()
        timeframe = self.timeframe_combo.currentText()
        days = self.days_spin.value()
        
        self.status_bar.showMessage(f"Cargando datos para {symbol} {timeframe}...")
        
        # Usar un hilo para cargar datos sin bloquear la UI
        self.data_worker = DataWorker(symbol, timeframe, days)
        self.data_worker.data_loaded.connect(self.on_data_loaded)
        self.data_worker.error_occurred.connect(self.on_data_error)
        self.data_worker.start()
    
    def on_data_loaded(self, data):
        """Cuando los datos se cargan correctamente"""
        self.status_bar.showMessage("Datos cargados correctamente")
        
        # Actualizar la tabla
        self.update_data_table(data)
    
    def on_data_error(self, error_msg):
        """Cuando ocurre un error cargando datos"""
        self.status_bar.showMessage("Error cargando datos")
        QMessageBox.critical(self, "Error", error_msg)
    
    def update_data_table(self, data):
        """Actualizar la tabla con los datos cargados"""
        self.data_table.setRowCount(len(data))
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels(["Fecha", "Open", "High", "Low", "Close", "Volume"])
        
        for row, (idx, values) in enumerate(data.iterrows()):
            self.data_table.setItem(row, 0, QTableWidgetItem(idx.strftime("%Y-%m-%d %H:%M")))
            self.data_table.setItem(row, 1, QTableWidgetItem(str(values['open'])))
            self.data_table.setItem(row, 2, QTableWidgetItem(str(values['high'])))
            self.data_table.setItem(row, 3, QTableWidgetItem(str(values['low'])))
            self.data_table.setItem(row, 4, QTableWidgetItem(str(values['close'])))
            self.data_table.setItem(row, 5, QTableWidgetItem(str(values.get('volume', 0))))
        
        self.data_table.resizeColumnsToContents()
    
    def save_strategy(self):
        """Guardar estrategia en un archivo"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Estrategia", "", "Python Files (*.py)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.strategy_code_editor.toPlainText())
                self.status_bar.showMessage("Estrategia guardada correctamente")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo guardar la estrategia: {str(e)}")
    
    def load_strategy(self):
        """Cargar estrategia desde un archivo"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Cargar Estrategia", "", "Python Files (*.py)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.strategy_code_editor.setPlainText(f.read())
                self.status_bar.showMessage("Estrategia cargada correctamente")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo cargar la estrategia: {str(e)}")
    
    def test_strategy(self):
        """Probar la estrategia actual"""
        # En la Fase 2, haremos una prueba básica de sintaxis
        code = self.strategy_code_editor.toPlainText()
        if not code.strip():
            QMessageBox.warning(self, "Advertencia", "No hay código para probar")
            return
        
        try:
            # Intentar compilar el código para verificar sintaxis
            compile(code, '<string>', 'exec')
            self.status_bar.showMessage("Estrategia compilada correctamente")
            QMessageBox.information(self, "Prueba", "La estrategia tiene sintaxis válida")
        except SyntaxError as e:
            QMessageBox.critical(self, "Error de Sintaxis", f"Error en la estrategia: {str(e)}")
    
    def run_backtest(self):
        """Ejecutar backtest de la estrategia"""
        # En la Fase 2, haremos un backtest básico
        symbol = self.bt_symbol_combo.currentText()
        timeframe = self.bt_timeframe_combo.currentText()
        start_date = self.bt_start_date.text()
        end_date = self.bt_end_date.text()
        initial_capital = self.bt_initial_capital.value()
        
        if not start_date or not end_date:
            QMessageBox.warning(self, "Advertencia", "Por favor, ingresa fechas de inicio y fin")
            return
        
        try:
            # Cargar datos
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            platform = get_platform()
            data = platform.get_market_data(symbol, timeframe, days=(end - start).days)
            
            if data is None:
                QMessageBox.critical(self, "Error", "No se pudieron cargar los datos para el backtest")
                return
            
            # Filtrar por fechas
            data = data[(data.index >= start) & (data.index <= end)]
            
            # Ejecutar backtest básico (aquí integraríamos el motor de backtesting)
            results = self.run_basic_backtest(data, initial_capital)
            
            # Mostrar resultados
            self.display_backtest_results(results)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en el backtest: {str(e)}")
    
    def run_basic_backtest(self, data, initial_capital):
        """Backtest básico para demostración"""
        # Esto es un ejemplo muy simple. En la Fase 3 lo expandiremos.
        capital = initial_capital
        position = 0
        trades = []
        
        for i in range(1, len(data)):
            # Estrategia simple: Cruce de medias móviles
            if i > 20:
                current_close = data['close'].iloc[i]
                prev_close = data['close'].iloc[i-1]
                ma_short = data['close'].iloc[i-10:i].mean()
                ma_long = data['close'].iloc[i-20:i].mean()
                
                # Señal de compra
                if prev_close < ma_long and current_close > ma_long and position <= 0:
                    if position < 0:
                        # Cerrar corto
                        capital += position * (prev_close - current_close)
                        trades.append(('Cierre Corto', data.index[i], current_close, capital))
                        position = 0
                    
                    # Abrir largo
                    position = capital * 0.1 / current_close  # 10% del capital
                    trades.append(('Apertura Larga', data.index[i], current_close, capital))
                
                # Señal de venta
                elif prev_close > ma_long and current_close < ma_long and position >= 0:
                    if position > 0:
                        # Cerrar largo
                        capital += position * (current_close - prev_close)
                        trades.append(('Cierre Largo', data.index[i], current_close, capital))
                        position = 0
                    
                    # Abrir corto
                    position = -capital * 0.1 / current_close  # 10% del capital
                    trades.append(('Apertura Corta', data.index[i], current_close, capital))
        
        # Cerrar posición final
        if position != 0:
            capital += position * (data['close'].iloc[-1] - data['close'].iloc[-2])
            trades.append(('Cierre Final', data.index[-1], data['close'].iloc[-1], capital))
            position = 0
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': (capital - initial_capital) / initial_capital * 100,
            'total_trades': len(trades),
            'trades': trades
        }
    
    def display_backtest_results(self, results):
        """Mostrar resultados del backtest"""
        text = f"""
        RESULTADOS DEL BACKTEST
        ========================
        Capital Inicial: ${results['initial_capital']:,.2f}
        Capital Final: ${results['final_capital']:,.2f}
        Retorno Total: {results['total_return']:.2f}%
        Total Operaciones: {results['total_trades']}
        
        ÚLTIMAS 10 OPERACIONES:
        """
        
        for trade in results['trades'][-10:]:
            text += f"{trade[0]} - {trade[1].strftime('%Y-%m-%d')} - Precio: {trade[2]:.5f} - Capital: ${trade[3]:,.2f}\n"
        
        self.bt_results_text.setPlainText(text)
    
    def toggle_live_trading(self):
        """Iniciar/Detener trading en vivo"""
        # Por implementar en fases posteriores
        QMessageBox.information(self, "En Desarrollo", "Esta funcionalidad estará disponible en la Fase 3")
    
    def update_realtime_data(self):
        """Actualizar datos en tiempo real"""
        if self.platform.initialized:
            # Actualizar métricas de cuenta
            account_info = self.platform.get_account_summary()
            if account_info:
                self.balance_label.setText(f"${account_info.get('balance', 0):,.2f}")
                self.equity_label.setText(f"${account_info.get('equity', 0):,.2f}")
                self.margin_label.setText(f"${account_info.get('margin', 0):,.2f}")
                self.free_margin_label.setText(f"${account_info.get('free_margin', 0):,.2f}")

def run_gui():
    """Función para ejecutar la interfaz gráfica"""
    app = QApplication(sys.argv)
    
    # Establecer estilo moderno
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()