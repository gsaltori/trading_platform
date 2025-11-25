#!/usr/bin/env python3
"""
Interfaz completa con AUTOGENERACIÓN DE ESTRATEGIAS
La verdadera funcionalidad de la plataforma
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
from datetime import datetime
import json

class AdvancedTradingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Plataforma de Trading Algorítmico - Autogeneración de Estrategias")
        self.root.geometry("1400x900")
        
        # Queue para comunicación thread-safe
        self.log_queue = queue.Queue()
        
        self.platform = None
        self.generated_strategies = []
        self.ml_engine = None
        self.genetic_optimizer = None
        
        self.create_widgets()
        
    def create_widgets(self):
        """Crear interfaz completa"""
        
        # Frame principal con scrollbar
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Notebook (pestañas)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # PESTAÑA 1: AUTOGENERACIÓN DE ESTRATEGIAS ⭐
        tab_strategy_gen = ttk.Frame(self.notebook)
        self.notebook.add(tab_strategy_gen, text="🤖 Autogeneración de Estrategias")
        self.create_strategy_generation_tab(tab_strategy_gen)
        
        # PESTAÑA 2: Machine Learning
        tab_ml = ttk.Frame(self.notebook)
        self.notebook.add(tab_ml, text="🔮 Machine Learning")
        self.create_ml_tab(tab_ml)
        
        # PESTAÑA 3: Optimización Genética
        tab_optimization = ttk.Frame(self.notebook)
        self.notebook.add(tab_optimization, text="🧬 Optimización Genética")
        self.create_optimization_tab(tab_optimization)
        
        # PESTAÑA 4: Dashboard
        tab_dashboard = ttk.Frame(self.notebook)
        self.notebook.add(tab_dashboard, text="📊 Dashboard")
        self.create_dashboard_tab(tab_dashboard)
        
        # PESTAÑA 5: Backtesting
        tab_backtest = ttk.Frame(self.notebook)
        self.notebook.add(tab_backtest, text="📈 Backtesting")
        self.create_backtest_tab(tab_backtest)
        
        # PESTAÑA 6: Log
        tab_log = ttk.Frame(self.notebook)
        self.notebook.add(tab_log, text="📝 Log")
        self.create_log_tab(tab_log)
        
        # Barra de estado
        self.status_bar = ttk.Label(self.root, text="Listo - Conecta a MT5 para comenzar", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_strategy_generation_tab(self, parent):
        """⭐ PESTAÑA PRINCIPAL: Autogeneración de Estrategias"""
        
        # Frame principal con scroll
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Título
        title_frame = ttk.Frame(scrollable_frame)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title = ttk.Label(title_frame, text="🤖 AUTOGENERACIÓN INTELIGENTE DE ESTRATEGIAS", 
                         font=('Arial', 16, 'bold'))
        title.pack()
        
        subtitle = ttk.Label(title_frame, 
                            text="Genera estrategias óptimas usando Machine Learning + Optimización Genética",
                            font=('Arial', 10))
        subtitle.pack()
        
        # SECCIÓN 1: Configuración de datos
        data_frame = ttk.LabelFrame(scrollable_frame, text="📊 1. Configuración de Datos", padding=10)
        data_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Símbolos
        ttk.Label(data_frame, text="Símbolos (separados por coma):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.entry_symbols = ttk.Entry(data_frame, width=50)
        self.entry_symbols.insert(0, "EURUSD,GBPUSD,USDJPY")
        self.entry_symbols.grid(row=0, column=1, padx=5, pady=2)
        
        # Timeframe
        ttk.Label(data_frame, text="Timeframe:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.cmb_gen_timeframe = ttk.Combobox(data_frame, values=["M1", "M5", "M15", "M30", "H1", "H4", "D1"], width=15)
        self.cmb_gen_timeframe.current(4)  # H1
        self.cmb_gen_timeframe.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Días de datos
        ttk.Label(data_frame, text="Días de datos históricos:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.spin_days = ttk.Spinbox(data_frame, from_=30, to=730, width=15)
        self.spin_days.set(180)
        self.spin_days.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # SECCIÓN 2: Configuración de generación
        gen_frame = ttk.LabelFrame(scrollable_frame, text="🧬 2. Configuración de Generación", padding=10)
        gen_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Número de estrategias a generar
        ttk.Label(gen_frame, text="Número de estrategias a generar:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.spin_num_strategies = ttk.Spinbox(gen_frame, from_=1, to=20, width=15)
        self.spin_num_strategies.set(5)
        self.spin_num_strategies.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Tipos de estrategias base
        ttk.Label(gen_frame, text="Tipos de estrategias base:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.strategy_types_frame = ttk.Frame(gen_frame)
        self.strategy_types_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.var_ma = tk.BooleanVar(value=True)
        self.var_rsi = tk.BooleanVar(value=True)
        self.var_macd = tk.BooleanVar(value=True)
        self.var_ml = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(self.strategy_types_frame, text="MA Crossover", variable=self.var_ma).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(self.strategy_types_frame, text="RSI", variable=self.var_rsi).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(self.strategy_types_frame, text="MACD", variable=self.var_macd).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(self.strategy_types_frame, text="ML-Based", variable=self.var_ml).pack(side=tk.LEFT, padx=2)
        
        # Usar ML para optimización
        ttk.Label(gen_frame, text="Usar Machine Learning:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.var_use_ml = tk.BooleanVar(value=True)
        ttk.Checkbutton(gen_frame, text="Entrenar modelos ML para predicción", 
                       variable=self.var_use_ml).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Usar optimización genética
        ttk.Label(gen_frame, text="Usar Optimización Genética:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.var_use_genetic = tk.BooleanVar(value=True)
        ttk.Checkbutton(gen_frame, text="Optimizar parámetros con algoritmos genéticos", 
                       variable=self.var_use_genetic).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Objetivo de optimización
        ttk.Label(gen_frame, text="Objetivo de optimización:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.cmb_objective = ttk.Combobox(gen_frame, 
                                         values=["Sharpe Ratio", "Profit Factor", "Total Return", "Calmar Ratio"],
                                         width=20)
        self.cmb_objective.current(0)
        self.cmb_objective.grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        
        # SECCIÓN 3: Parámetros avanzados
        advanced_frame = ttk.LabelFrame(scrollable_frame, text="⚙️ 3. Parámetros Avanzados (Opcional)", padding=10)
        advanced_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Población para genético
        ttk.Label(advanced_frame, text="Población genética:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.spin_population = ttk.Spinbox(advanced_frame, from_=20, to=200, width=15)
        self.spin_population.set(50)
        self.spin_population.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Generaciones
        ttk.Label(advanced_frame, text="Generaciones:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.spin_generations = ttk.Spinbox(advanced_frame, from_=10, to=100, width=15)
        self.spin_generations.set(30)
        self.spin_generations.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Win rate mínimo
        ttk.Label(advanced_frame, text="Win rate mínimo (%):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.spin_min_winrate = ttk.Spinbox(advanced_frame, from_=30, to=80, width=15)
        self.spin_min_winrate.set(50)
        self.spin_min_winrate.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Botón principal de generación
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=15)
        
        self.btn_generate = ttk.Button(btn_frame, text="🚀 GENERAR ESTRATEGIAS", 
                                       command=self.generate_strategies,
                                       style='Accent.TButton')
        self.btn_generate.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="⏹️ Detener", command=self.stop_generation).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="💾 Guardar Estrategias", command=self.save_strategies).pack(side=tk.LEFT, padx=5)
        
        # Barra de progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(scrollable_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        self.lbl_progress = ttk.Label(scrollable_frame, text="Esperando...")
        self.lbl_progress.pack(pady=2)
        
        # SECCIÓN 4: Resultados
        results_frame = ttk.LabelFrame(scrollable_frame, text="📊 4. Estrategias Generadas", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tabla de estrategias
        columns = ('Nombre', 'Tipo', 'Sharpe', 'Win Rate', 'Return', 'Drawdown', 'Profit Factor')
        self.tree_strategies = ttk.Treeview(results_frame, columns=columns, show='tree headings', height=8)
        
        self.tree_strategies.heading('#0', text='ID')
        self.tree_strategies.column('#0', width=50)
        
        for col in columns:
            self.tree_strategies.heading(col, text=col)
            self.tree_strategies.column(col, width=120)
        
        # Scrollbar para la tabla
        tree_scroll = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree_strategies.yview)
        self.tree_strategies.configure(yscrollcommand=tree_scroll.set)
        
        self.tree_strategies.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Botones de acción para estrategias
        action_frame = ttk.Frame(scrollable_frame)
        action_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(action_frame, text="📈 Backtest Detallado", 
                  command=self.detailed_backtest_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="🚀 Activar en Vivo", 
                  command=self.activate_strategy_live).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="📄 Ver Código", 
                  command=self.view_strategy_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="🔄 Refinar Estrategia", 
                  command=self.refine_strategy).pack(side=tk.LEFT, padx=2)
        
        # Empaquetar canvas y scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_ml_tab(self, parent):
        """Pestaña de Machine Learning"""
        
        title = ttk.Label(parent, text="🔮 Machine Learning Engine", font=('Arial', 14, 'bold'))
        title.pack(pady=10)
        
        # Frame de configuración
        config_frame = ttk.LabelFrame(parent, text="Configuración del Modelo", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(config_frame, text="Algoritmo:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.cmb_ml_algo = ttk.Combobox(config_frame, 
                                        values=["XGBoost", "Random Forest", "LSTM", "Ensemble"],
                                        width=20)
        self.cmb_ml_algo.current(0)
        self.cmb_ml_algo.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(config_frame, text="Tipo de predicción:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.cmb_ml_type = ttk.Combobox(config_frame,
                                        values=["Clasificación (Dirección)", "Regresión (Precio)"],
                                        width=20)
        self.cmb_ml_type.current(0)
        self.cmb_ml_type.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Button(config_frame, text="🎓 Entrenar Modelo", 
                  command=self.train_ml_model).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Resultados
        results_frame = ttk.LabelFrame(parent, text="Resultados del Entrenamiento", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.txt_ml_results = scrolledtext.ScrolledText(results_frame, height=20)
        self.txt_ml_results.pack(fill=tk.BOTH, expand=True)
        
    def create_optimization_tab(self, parent):
        """Pestaña de Optimización Genética"""
        
        title = ttk.Label(parent, text="🧬 Optimización Genética", font=('Arial', 14, 'bold'))
        title.pack(pady=10)
        
        # Frame de configuración
        config_frame = ttk.LabelFrame(parent, text="Parámetros de Optimización", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(config_frame, text="Estrategia base:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.cmb_opt_strategy = ttk.Combobox(config_frame,
                                             values=["MA Crossover", "RSI", "MACD"],
                                             width=20)
        self.cmb_opt_strategy.current(0)
        self.cmb_opt_strategy.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Button(config_frame, text="🔬 Optimizar", 
                  command=self.optimize_strategy).grid(row=1, column=0, columnspan=2, pady=10)
        
        # Resultados
        results_frame = ttk.LabelFrame(parent, text="Resultados de Optimización", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.txt_opt_results = scrolledtext.ScrolledText(results_frame, height=20)
        self.txt_opt_results.pack(fill=tk.BOTH, expand=True)
        
    def create_dashboard_tab(self, parent):
        """Dashboard básico"""
        
        title = ttk.Label(parent, text="📊 Dashboard", font=('Arial', 14, 'bold'))
        title.pack(pady=10)
        
        # Conexión MT5
        conn_frame = ttk.LabelFrame(parent, text="Conexión MT5", padding=10)
        conn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_connect = ttk.Button(conn_frame, text="🔌 Conectar MT5", command=self.connect_mt5)
        self.btn_connect.pack(side=tk.LEFT, padx=5)
        
        self.lbl_connection = ttk.Label(conn_frame, text="Desconectado", foreground="red")
        self.lbl_connection.pack(side=tk.LEFT, padx=5)
        
        # Métricas
        metrics_frame = ttk.LabelFrame(parent, text="Métricas de Cuenta", padding=10)
        metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(metrics_frame, text="Balance:").grid(row=0, column=0, sticky=tk.W)
        self.lbl_balance = ttk.Label(metrics_frame, text="--", font=('Arial', 12, 'bold'))
        self.lbl_balance.grid(row=0, column=1, sticky=tk.W, padx=10)
        
        ttk.Label(metrics_frame, text="Equity:").grid(row=1, column=0, sticky=tk.W)
        self.lbl_equity = ttk.Label(metrics_frame, text="--", font=('Arial', 12, 'bold'))
        self.lbl_equity.grid(row=1, column=1, sticky=tk.W, padx=10)
        
    def create_backtest_tab(self, parent):
        """Pestaña de backtesting individual"""
        
        title = ttk.Label(parent, text="📈 Backtesting", font=('Arial', 14, 'bold'))
        title.pack(pady=10)
        
        # Controles
        controls_frame = ttk.LabelFrame(parent, text="Parámetros", padding=10)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(controls_frame, text="Símbolo:").grid(row=0, column=0)
        self.cmb_symbol = ttk.Combobox(controls_frame, values=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
        self.cmb_symbol.current(0)
        self.cmb_symbol.grid(row=0, column=1, padx=5)
        
        ttk.Label(controls_frame, text="Timeframe:").grid(row=0, column=2)
        self.cmb_timeframe = ttk.Combobox(controls_frame, values=["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
        self.cmb_timeframe.current(4)
        self.cmb_timeframe.grid(row=0, column=3, padx=5)
        
        ttk.Button(controls_frame, text="▶️ Ejecutar Backtest", 
                  command=self.run_single_backtest).grid(row=1, column=0, columnspan=4, pady=10)
        
        # Resultados
        results_frame = ttk.LabelFrame(parent, text="Resultados", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.txt_backtest_results = scrolledtext.ScrolledText(results_frame, height=20)
        self.txt_backtest_results.pack(fill=tk.BOTH, expand=True)
        
    def create_log_tab(self, parent):
        """Pestaña de log"""
        
        self.txt_log = scrolledtext.ScrolledText(parent, height=30, width=100, font=('Consolas', 9))
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Button(parent, text="🗑️ Limpiar Log", 
                  command=lambda: self.txt_log.delete(1.0, tk.END)).pack(pady=5)
        
    def log(self, message, level="INFO"):
        """Agregar mensaje al log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color según nivel
        if level == "ERROR":
            prefix = "❌"
        elif level == "SUCCESS":
            prefix = "✅"
        elif level == "WARNING":
            prefix = "⚠️"
        else:
            prefix = "ℹ️"
        
        log_message = f"[{timestamp}] {prefix} {message}\n"
        self.txt_log.insert(tk.END, log_message)
        self.txt_log.see(tk.END)
        self.root.update_idletasks()
        
    def update_status(self, message):
        """Actualizar barra de estado"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
        
    def connect_mt5(self):
        """Conectar a MT5"""
        self.log("Conectando a MT5...")
        
        def connect_thread():
            try:
                from core.platform import get_platform
                self.platform = get_platform()
                
                if self.platform.initialize():
                    self.root.after(0, lambda: self.lbl_connection.config(text="✅ Conectado", foreground="green"))
                    self.log("Conectado a MT5 exitosamente", "SUCCESS")
                    self.update_account_info()
                    self.update_status("MT5 Conectado - Listo para generar estrategias")
                else:
                    self.log("Error conectando a MT5", "ERROR")
            except Exception as e:
                self.log(f"Error: {e}", "ERROR")
        
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
            self.root.after(5000, self.update_account_info)
        except:
            pass
    
    def generate_strategies(self):
        """🚀 FUNCIÓN PRINCIPAL: Generar estrategias automáticamente"""
        self.log("="*50, "INFO")
        self.log("INICIANDO AUTOGENERACIÓN DE ESTRATEGIAS", "INFO")
        self.log("="*50, "INFO")
        
        # Validar conexión
        if not self.platform or not self.platform.initialized:
            messagebox.showerror("Error", "Primero conecta a MT5")
            return
        
        # Deshabilitar botón
        self.btn_generate.config(state='disabled')
        self.progress_var.set(0)
        
        def generation_thread():
            try:
                # Obtener configuración
                symbols = [s.strip() for s in self.entry_symbols.get().split(',')]
                timeframe = self.cmb_gen_timeframe.get()
                days = int(self.spin_days.get())
                num_strategies = int(self.spin_num_strategies.get())
                
                self.log(f"Símbolos: {', '.join(symbols)}")
                self.log(f"Timeframe: {timeframe}, Días: {days}")
                self.log(f"Generando {num_strategies} estrategias...")
                
                self.root.after(0, lambda: self.update_status("Paso 1/5: Obteniendo datos..."))
                self.root.after(0, lambda: self.lbl_progress.config(text="Obteniendo datos históricos..."))
                
                # Paso 1: Obtener datos (continuación en el siguiente mensaje debido a límite de espacio)
                
                data_dict = {}
                for i, symbol in enumerate(symbols):
                    progress = (i + 1) / len(symbols) * 20  # 20% del progreso total
                    self.root.after(0, lambda p=progress: self.progress_var.set(p))
                    
                    self.log(f"Descargando datos de {symbol}...")
                    data = self.platform.get_market_data(symbol, timeframe, days=days)
                    
                    if data is not None and len(data) > 100:
                        data_dict[symbol] = data
                        self.log(f"✓ {symbol}: {len(data)} velas", "SUCCESS")
                    else:
                        self.log(f"✗ {symbol}: Datos insuficientes", "WARNING")
                
                if not data_dict:
                    self.log("Error: No se pudieron obtener datos", "ERROR")
                    return
                
                # Paso 2: Machine Learning (si está habilitado)
                ml_models = {}
                if self.var_use_ml.get():
                    self.root.after(0, lambda: self.update_status("Paso 2/5: Entrenando modelos ML..."))
                    self.root.after(0, lambda: self.lbl_progress.config(text="Entrenando Machine Learning..."))
                    
                    from ml.ml_engine import MLEngine, MLModelConfig
                    ml_engine = MLEngine()
                    
                    for symbol in data_dict.keys():
                        self.log(f"Entrenando modelo ML para {symbol}...")
                        
                        config = MLModelConfig(
                            model_type='classification',
                            algorithm='xgboost',
                            features=[],
                            target='price_direction',
                            parameters={'n_estimators': 100, 'max_depth': 6}
                        )
                        
                        try:
                            result = ml_engine.train_model(data_dict[symbol], config)
                            ml_models[symbol] = result
                            self.log(f"✓ {symbol} ML: Accuracy {result.metrics['accuracy']:.3f}", "SUCCESS")
                        except Exception as e:
                            self.log(f"✗ {symbol} ML: {str(e)[:50]}", "WARNING")
                    
                    self.root.after(0, lambda: self.progress_var.set(40))
                
                # Paso 3: Generar estrategias base
                self.root.after(0, lambda: self.update_status("Paso 3/5: Generando estrategias base..."))
                self.root.after(0, lambda: self.lbl_progress.config(text="Generando estrategias..."))
                
                from strategies.strategy_engine import StrategyEngine, StrategyConfig
                strategy_engine = StrategyEngine()
                
                generated = []
                strategy_types = []
                if self.var_ma.get():
                    strategy_types.append('ma_crossover')
                if self.var_rsi.get():
                    strategy_types.append('rsi')
                if self.var_macd.get():
                    strategy_types.append('macd')
                
                # Generar combinaciones
                import random
                for i in range(num_strategies):
                    strategy_type = random.choice(strategy_types)
                    symbol = random.choice(list(data_dict.keys()))
                    
                    # Parámetros aleatorios
                    if strategy_type == 'ma_crossover':
                        params = {
                            'fast_period': random.randint(5, 20),
                            'slow_period': random.randint(20, 50),
                            'rsi_period': random.randint(10, 20)
                        }
                    elif strategy_type == 'rsi':
                        params = {
                            'rsi_period': random.randint(10, 20),
                            'rsi_oversold': random.randint(20, 35),
                            'rsi_overbought': random.randint(65, 80)
                        }
                    else:  # macd
                        params = {
                            'fast_period': random.randint(8, 15),
                            'slow_period': random.randint(20, 30),
                            'signal_period': random.randint(7, 12)
                        }
                    
                    config = StrategyConfig(
                        name=f"AutoGen_{strategy_type}_{i+1}",
                        symbols=[symbol],
                        timeframe=timeframe,
                        parameters=params
                    )
                    
                    strategy = strategy_engine.create_strategy(strategy_type, config)
                    generated.append((strategy, symbol, strategy_type))
                
                self.log(f"✓ Generadas {len(generated)} estrategias base", "SUCCESS")
                self.root.after(0, lambda: self.progress_var.set(60))
                
                # Paso 4: Optimización genética (si está habilitada)
                if self.var_use_genetic.get():
                    self.root.after(0, lambda: self.update_status("Paso 4/5: Optimizando con algoritmos genéticos..."))
                    self.root.after(0, lambda: self.lbl_progress.config(text="Optimización genética en progreso..."))
                    
                    from optimization.genetic_optimizer import GeneticOptimizer, OptimizationConfig
                    from backtesting.backtest_engine import BacktestEngine
                    
                    backtest_engine = BacktestEngine()
                    genetic_optimizer = GeneticOptimizer(backtest_engine)
                    
                    for i, (strategy, symbol, strategy_type) in enumerate(generated[:3]):  # Optimizar solo las primeras 3
                        self.log(f"Optimizando {strategy.name}...")
                        
                        # Definir rangos según tipo
                        if strategy_type == 'ma_crossover':
                            param_ranges = {
                                'fast_period_int': (5, 20),
                                'slow_period_int': (20, 50)
                            }
                        elif strategy_type == 'rsi':
                            param_ranges = {
                                'rsi_period_int': (10, 20),
                                'rsi_oversold_int': (20, 35),
                                'rsi_overbought_int': (65, 80)
                            }
                        else:
                            param_ranges = {
                                'fast_period_int': (8, 15),
                                'slow_period_int': (20, 30)
                            }
                        
                        opt_config = OptimizationConfig(
                            strategy_name=strategy.name,
                            parameter_ranges=param_ranges,
                            objective='sharpe',
                            population_size=int(self.spin_population.get()),
                            generations=int(self.spin_generations.get())
                        )
                        
                        try:
                            result = genetic_optimizer.optimize_strategy(strategy, data_dict[symbol], opt_config)
                            strategy.config.parameters.update(result['best_parameters'])
                            self.log(f"✓ {strategy.name} optimizada: Sharpe {result['best_fitness']:.2f}", "SUCCESS")
                        except Exception as e:
                            self.log(f"✗ Error optimizando {strategy.name}: {str(e)[:50]}", "WARNING")
                    
                    self.root.after(0, lambda: self.progress_var.set(80))
                
                # Paso 5: Backtest final y ranking
                self.root.after(0, lambda: self.update_status("Paso 5/5: Evaluando estrategias..."))
                self.root.after(0, lambda: self.lbl_progress.config(text="Evaluación final..."))
                
                from backtesting.backtest_engine import BacktestEngine
                backtest_engine = BacktestEngine(initial_capital=10000)
                
                final_results = []
                for strategy, symbol, strategy_type in generated:
                    try:
                        result = backtest_engine.run_backtest(
                            data=data_dict[symbol],
                            strategy=strategy,
                            symbol=symbol,
                            commission=0.001
                        )
                        
                        # Filtrar por win rate mínimo
                        min_winrate = float(self.spin_min_winrate.get())
                        if result.win_rate >= min_winrate and result.total_trades >= 10:
                            final_results.append({
                                'strategy': strategy,
                                'symbol': symbol,
                                'type': strategy_type,
                                'result': result
                            })
                    except Exception as e:
                        self.log(f"Error en backtest de {strategy.name}: {str(e)[:50]}", "WARNING")
                
                # Ordenar por Sharpe Ratio
                final_results.sort(key=lambda x: x['result'].sharpe_ratio or -999, reverse=True)
                
                # Mostrar resultados
                self.root.after(0, lambda: self.tree_strategies.delete(*self.tree_strategies.get_children()))
                
                for i, item in enumerate(final_results[:num_strategies]):
                    result = item['result']
                    sharpe_str = f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio else "N/A"
                    pf_str = f"{result.profit_factor:.2f}" if result.profit_factor else "N/A"
                    
                    self.root.after(0, lambda idx=i, itm=item, s=sharpe_str, p=pf_str: 
                        self.tree_strategies.insert('', 'end', text=str(idx+1), values=(
                            itm['strategy'].name,
                            itm['type'].upper(),
                            s,
                            f"{itm['result'].win_rate:.1f}%",
                            f"{itm['result'].total_return:.2f}%",
                            f"{itm['result'].max_drawdown:.2f}%",
                            p
                        ))
                    )
                
                self.generated_strategies = final_results
                
                # Finalizar
                self.root.after(0, lambda: self.progress_var.set(100))
                self.root.after(0, lambda: self.lbl_progress.config(text="✅ ¡Completado!"))
                self.root.after(0, lambda: self.update_status(f"✅ Generadas {len(final_results)} estrategias exitosamente"))
                
                self.log("="*50, "INFO")
                self.log(f"✅ AUTOGENERACIÓN COMPLETADA", "SUCCESS")
                self.log(f"Total de estrategias viables: {len(final_results)}", "SUCCESS")
                self.log("="*50, "INFO")
                
                # Mostrar resumen
                if final_results:
                    best = final_results[0]
                    self.log(f"🏆 Mejor estrategia: {best['strategy'].name}", "SUCCESS")
                    self.log(f"   Sharpe Ratio: {best['result'].sharpe_ratio:.2f}" if best['result'].sharpe_ratio else "   Sharpe: N/A", "INFO")
                    self.log(f"   Win Rate: {best['result'].win_rate:.1f}%", "INFO")
                    self.log(f"   Total Return: {best['result'].total_return:.2f}%", "INFO")
                
            except Exception as e:
                self.log(f"ERROR CRÍTICO: {e}", "ERROR")
                import traceback
                self.log(traceback.format_exc(), "ERROR")
            
            finally:
                self.root.after(0, lambda: self.btn_generate.config(state='normal'))
        
        thread = threading.Thread(target=generation_thread, daemon=True)
        thread.start()
    
    def stop_generation(self):
        """Detener generación"""
        self.log("Deteniendo generación...", "WARNING")
        # Implementar lógica de detención
    
    def save_strategies(self):
        """Guardar estrategias generadas"""
        if not self.generated_strategies:
            messagebox.showwarning("Advertencia", "No hay estrategias para guardar")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            strategies_data = []
            for item in self.generated_strategies:
                strategies_data.append({
                    'name': item['strategy'].name,
                    'type': item['type'],
                    'symbol': item['symbol'],
                    'parameters': item['strategy'].config.parameters,
                    'sharpe_ratio': item['result'].sharpe_ratio,
                    'win_rate': item['result'].win_rate,
                    'total_return': item['result'].total_return
                })
            
            with open(filename, 'w') as f:
                json.dump(strategies_data, f, indent=2)
            
            self.log(f"✓ Estrategias guardadas en {filename}", "SUCCESS")
            messagebox.showinfo("Éxito", f"Estrategias guardadas en:\n{filename}")
    
    def detailed_backtest_selected(self):
        """Backtest detallado de estrategia seleccionada"""
        selection = self.tree_strategies.selection()
        if not selection:
            messagebox.showwarning("Advertencia", "Selecciona una estrategia")
            return
        
        index = int(self.tree_strategies.item(selection[0])['text']) - 1
        if index < len(self.generated_strategies):
            item = self.generated_strategies[index]
            self.log(f"Ejecutando backtest detallado de {item['strategy'].name}...", "INFO")
            # Implementar backtest detallado
    
    def activate_strategy_live(self):
        """Activar estrategia para trading en vivo"""
        selection = self.tree_strategies.selection()
        if not selection:
            messagebox.showwarning("Advertencia", "Selecciona una estrategia")
            return
        
        response = messagebox.askyesno("Confirmar", 
                                      "¿Activar esta estrategia para trading en vivo?\n\n⚠️ Esto usará dinero real.")
        if response:
            self.log("Activando estrategia para trading en vivo...", "WARNING")
            # Implementar activación en vivo
    
    def view_strategy_code(self):
        """Ver código de la estrategia"""
        selection = self.tree_strategies.selection()
        if not selection:
            messagebox.showwarning("Advertencia", "Selecciona una estrategia")
            return
        
        index = int(self.tree_strategies.item(selection[0])['text']) - 1
        if index < len(self.generated_strategies):
            item = self.generated_strategies[index]
            
            code_window = tk.Toplevel(self.root)
            code_window.title(f"Código: {item['strategy'].name}")
            code_window.geometry("800x600")
            
            text = scrolledtext.ScrolledText(code_window, font=('Consolas', 10))
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Mostrar código de la estrategia
            code = f"""
# Estrategia: {item['strategy'].name}
# Tipo: {item['type']}
# Símbolo: {item['symbol']}

# Parámetros:
{json.dumps(item['strategy'].config.parameters, indent=2)}

# Métricas de Backtest:
# - Sharpe Ratio: {item['result'].sharpe_ratio:.2f if item['result'].sharpe_ratio else 'N/A'}
# - Win Rate: {item['result'].win_rate:.1f}%
# - Total Return: {item['result'].total_return:.2f}%
# - Max Drawdown: {item['result'].max_drawdown:.2f}%
# - Total Trades: {item['result'].total_trades}
"""
            text.insert('1.0', code)
            text.config(state='disabled')
    
    def refine_strategy(self):
        """Refinar estrategia seleccionada"""
        selection = self.tree_strategies.selection()
        if not selection:
            messagebox.showwarning("Advertencia", "Selecciona una estrategia")
            return
        
        self.log("Refinando estrategia...", "INFO")
        # Implementar refinamiento
    
    def train_ml_model(self):
        """Entrenar modelo ML individual"""
        self.log("Entrenando modelo ML...", "INFO")
        # Implementar entrenamiento ML
    
    def optimize_strategy(self):
        """Optimizar estrategia individual"""
        self.log("Optimizando estrategia...", "INFO")
        # Implementar optimización
    
    def run_single_backtest(self):
        """Ejecutar backtest individual"""
        self.log("Ejecutando backtest...", "INFO")
        # Implementar backtest individual (ya lo tienes en la versión anterior)
    
    def run(self):
        """Ejecutar GUI"""
        self.log("🚀 Plataforma de Autogeneración de Estrategias iniciada", "SUCCESS")
        self.log("Conecta a MT5 y ve a la pestaña '🤖 Autogeneración de Estrategias'", "INFO")
        self.root.mainloop()

def run_advanced_gui():
    """Función para ejecutar desde main.py"""
    app = AdvancedTradingGUI()
    app.run()

if __name__ == "__main__":
    print("🚀 Iniciando Plataforma de Autogeneración de Estrategias")
    print("="*70)
    print("Funcionalidades:")
    print("  ✅ Autogeneración inteligente de estrategias")
    print("  ✅ Machine Learning integrado")
    print("  ✅ Optimización genética")
    print("  ✅ Backtesting automático")
    print("  ✅ Ranking y selección")
    print("="*70)
    
    app = AdvancedTradingGUI()
    app.run()