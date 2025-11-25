#!/usr/bin/env python3
"""
Interfaz alternativa simple usando tkinter (viene incluido con Python)
No requiere PyQt6 ni Visual C++ Redistributable
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
from datetime import datetime

class SimpleTradingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Plataforma de Trading - Interfaz Simple")
        self.root.geometry("1200x800")
        
        # Queue para comunicación thread-safe
        self.log_queue = queue.Queue()
        
        self.create_widgets()
        self.platform = None
        
    def create_widgets(self):
        """Crear interfaz"""
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Notebook (pestañas)
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Pestaña 1: Dashboard
        tab_dashboard = ttk.Frame(notebook)
        notebook.add(tab_dashboard, text="Dashboard")
        self.create_dashboard_tab(tab_dashboard)
        
        # Pestaña 2: Backtesting
        tab_backtest = ttk.Frame(notebook)
        notebook.add(tab_backtest, text="Backtesting")
        self.create_backtest_tab(tab_backtest)
        
        # Pestaña 3: Log
        tab_log = ttk.Frame(notebook)
        notebook.add(tab_log, text="Log")
        self.create_log_tab(tab_log)
        
        # Barra de estado
        self.status_bar = ttk.Label(self.root, text="Listo", relief=tk.SUNKEN)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Configurar pesos para redimensionamiento
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
    def create_dashboard_tab(self, parent):
        """Crear dashboard"""
        
        # Título
        title = ttk.Label(parent, text="Dashboard Principal", font=('Arial', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Frame de conexión
        conn_frame = ttk.LabelFrame(parent, text="Conexión MT5", padding="10")
        conn_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.btn_connect = ttk.Button(conn_frame, text="Conectar MT5", command=self.connect_mt5)
        self.btn_connect.pack(side=tk.LEFT, padx=5)
        
        self.lbl_connection = ttk.Label(conn_frame, text="Desconectado", foreground="red")
        self.lbl_connection.pack(side=tk.LEFT, padx=5)
        
        # Frame de métricas
        metrics_frame = ttk.LabelFrame(parent, text="Métricas de Cuenta", padding="10")
        metrics_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(metrics_frame, text="Balance:").grid(row=0, column=0, sticky=tk.W)
        self.lbl_balance = ttk.Label(metrics_frame, text="--")
        self.lbl_balance.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(metrics_frame, text="Equity:").grid(row=1, column=0, sticky=tk.W)
        self.lbl_equity = ttk.Label(metrics_frame, text="--")
        self.lbl_equity.grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(metrics_frame, text="Margen:").grid(row=2, column=0, sticky=tk.W)
        self.lbl_margin = ttk.Label(metrics_frame, text="--")
        self.lbl_margin.grid(row=2, column=1, sticky=tk.W)
        
    def create_backtest_tab(self, parent):
        """Crear pestaña de backtesting"""
        
        # Controles
        controls_frame = ttk.LabelFrame(parent, text="Parámetros", padding="10")
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(controls_frame, text="Símbolo:").grid(row=0, column=0)
        self.cmb_symbol = ttk.Combobox(controls_frame, values=["EURUSD", "GBPUSD", "USDJPY"])
        self.cmb_symbol.current(0)
        self.cmb_symbol.grid(row=0, column=1)
        
        ttk.Label(controls_frame, text="Timeframe:").grid(row=1, column=0)
        self.cmb_timeframe = ttk.Combobox(controls_frame, values=["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
        self.cmb_timeframe.current(4)
        self.cmb_timeframe.grid(row=1, column=1)
        
        ttk.Button(controls_frame, text="Ejecutar Backtest", command=self.run_backtest).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Resultados
        results_frame = ttk.LabelFrame(parent, text="Resultados", padding="10")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.txt_backtest_results = scrolledtext.ScrolledText(results_frame, height=20, width=80)
        self.txt_backtest_results.pack(expand=True, fill=tk.BOTH)
        
    def create_log_tab(self, parent):
        """Crear pestaña de log"""
        
        self.txt_log = scrolledtext.ScrolledText(parent, height=30, width=100)
        self.txt_log.pack(expand=True, fill=tk.BOTH)
        
        # Botón para limpiar log
        ttk.Button(parent, text="Limpiar Log", command=lambda: self.txt_log.delete(1.0, tk.END)).pack(pady=5)
        
    def log(self, message):
        """Agregar mensaje al log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.txt_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.txt_log.see(tk.END)
        
    def connect_mt5(self):
        """Conectar a MT5"""
        self.log("Conectando a MT5...")
        
        def connect_thread():
            try:
                from core.platform import get_platform
                self.platform = get_platform()
                
                if self.platform.initialize():
                    self.root.after(0, lambda: self.lbl_connection.config(text="Conectado", foreground="green"))
                    self.log("✓ Conectado a MT5 exitosamente")
                    self.update_account_info()
                else:
                    self.log("✗ Error conectando a MT5")
                    messagebox.showerror("Error", "No se pudo conectar a MT5")
            except Exception as e:
                self.log(f"✗ Error: {e}")
                messagebox.showerror("Error", f"Error conectando: {e}")
        
        thread = threading.Thread(target=connect_thread, daemon=True)
        thread.start()
        
    def update_account_info(self):
        """Actualizar información de cuenta"""
        if not self.platform or not self.platform.initialized:
            return
        
        try:
            account_info = self.platform.get_account_summary()
            
            self.lbl_balance.config(text=f"${account_info.get('balance', 0):,.2f}")
            self.lbl_equity.config(text=f"${account_info.get('equity', 0):,.2f}")
            self.lbl_margin.config(text=f"${account_info.get('margin', 0):,.2f}")
            
            # Actualizar cada 5 segundos
            self.root.after(5000, self.update_account_info)
        except Exception as e:
            self.log(f"Error actualizando info: {e}")
    
    def run_backtest(self):
        """Ejecutar backtest"""
        symbol = self.cmb_symbol.get()
        timeframe = self.cmb_timeframe.get()
        
        self.log(f"Iniciando backtest: {symbol} {timeframe}")
        self.txt_backtest_results.delete(1.0, tk.END)
        self.txt_backtest_results.insert(tk.END, "Ejecutando backtest...\n")
        
        def backtest_thread():
            try:
                if not self.platform or not self.platform.initialized:
                    from core.platform import get_platform
                    self.platform = get_platform()
                    self.platform.initialize()
                
                # Obtener datos
                self.log("Obteniendo datos...")
                data = self.platform.get_market_data(symbol, timeframe, days=90)
                
                if data is None:
                    self.log("✗ Error obteniendo datos")
                    return
                
                self.log(f"✓ Datos obtenidos: {len(data)} velas")
                
                # Crear estrategia simple para demo
                from strategies.strategy_engine import StrategyEngine, StrategyConfig
                
                engine = StrategyEngine()
                config = StrategyConfig(
                    name="Demo MA",
                    symbols=[symbol],
                    timeframe=timeframe,
                    parameters={'fast_period': 10, 'slow_period': 20}
                )
                
                strategy = engine.create_strategy('ma_crossover', config)
                self.log("✓ Estrategia creada")
                
                # Ejecutar backtest
                from backtesting.backtest_engine import BacktestEngine
                
                backtest = BacktestEngine(initial_capital=10000)
                result = backtest.run_backtest(data, strategy, symbol)
                
                # Mostrar resultados
                sharpe_str = f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio else "N/A"
                profit_factor_str = f"{result.profit_factor:.2f}" if result.profit_factor else "N/A"
                
                results_text = f"""
RESULTADOS DEL BACKTEST
{'='*50}

Estrategia: {result.strategy_name}
Símbolo: {symbol}
Timeframe: {timeframe}

RENDIMIENTO:
  Retorno Total: {result.total_return:.2f}%
  Total Trades: {result.total_trades}
  Win Rate: {result.win_rate:.1f}%
  
MÉTRICAS:
  Sharpe Ratio: {sharpe_str}
  Max Drawdown: {result.max_drawdown:.2f}%
  Profit Factor: {profit_factor_str}
  
TRADES:
  Ganadores: {result.winning_trades}
  Perdedores: {result.losing_trades}
  Avg Win: ${result.avg_winning_trade:.2f}
  Avg Loss: ${result.avg_losing_trade:.2f}
                """
                
                self.root.after(0, lambda: self.txt_backtest_results.delete(1.0, tk.END))
                self.root.after(0, lambda: self.txt_backtest_results.insert(tk.END, results_text))
                self.log("✓ Backtest completado")
                
            except Exception as e:
                error_msg = f"✗ Error en backtest: {e}"
                self.log(error_msg)
                self.root.after(0, lambda: self.txt_backtest_results.insert(tk.END, f"\nERROR: {e}"))
        
        thread = threading.Thread(target=backtest_thread, daemon=True)
        thread.start()
    
    def run(self):
        """Ejecutar GUI"""
        self.log("Interfaz iniciada - Versión Tkinter (sin PyQt6)")
        self.log("Conecta a MT5 para comenzar")
        self.root.mainloop()

def run_simple_gui():
    """Función para ejecutar desde main.py"""
    app = SimpleTradingGUI()
    app.run()

if __name__ == "__main__":
    print("🚀 Iniciando interfaz simple (Tkinter)")
    print("Esta versión NO requiere PyQt6")
    print("-"*50)
    
    app = SimpleTradingGUI()
    app.run()