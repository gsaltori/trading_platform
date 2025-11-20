# optimization/genetic_optimizer.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from deap import base, creator, tools, algorithms
import multiprocessing as mp

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    strategy_name: str
    parameter_ranges: Dict[str, Tuple[float, float]]
    objective: str  # 'sharpe', 'profit_factor', 'total_return', 'custom'
    population_size: int = 100
    generations: int = 50
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2
    tournament_size: int = 3
    hall_of_fame_size: int = 10
    n_workers: int = -1

class GeneticOptimizer:
    """Optimizador genético mejorado para estrategias de trading"""
    
    def __init__(self, backtest_engine):
        self.backtest_engine = backtest_engine
        self.hall_of_fame = []
        self.optimization_history = []
        
    def optimize_strategy(self, strategy, data: pd.DataFrame, 
                         config: OptimizationConfig) -> Dict[str, Any]:
        """Optimizar estrategia usando algoritmo genético"""
        logger.info(f"Iniciando optimización genética para {config.strategy_name}")
        
        # Configurar DEAP
        self._setup_deap(config.objective)
        
        # Crear toolbox
        toolbox = base.Toolbox()
        
        # Registrar funciones
        self._register_functions(toolbox, strategy, data, config)
        
        # Crear población inicial
        population = toolbox.population(n=config.population_size)
        
        # Estadísticas
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Hall of Fame
        hof = tools.HallOfFame(config.hall_of_fame_size)
        
        # Ejecutar algoritmo genético
        population, logbook = algorithms.eaSimple(
            population, toolbox,
            cxpb=config.crossover_prob,
            mutpb=config.mutation_prob,
            ngen=config.generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )
        
        # Guardar resultados
        best_individual = hof[0]
        best_params = self._decode_individual(best_individual, config.parameter_ranges)
        best_fitness = best_individual.fitness.values[0]
        
        result = {
            'best_parameters': best_params,
            'best_fitness': best_fitness,
            'hall_of_fame': [
                {
                    'parameters': self._decode_individual(ind, config.parameter_ranges),
                    'fitness': ind.fitness.values[0]
                }
                for ind in hof
            ],
            'optimization_history': logbook,
            'generations': config.generations,
            'population_size': config.population_size
        }
        
        logger.info(f"Optimización completada. Mejor fitness: {best_fitness:.4f}")
        return result
    
    def _setup_deap(self, objective: str):
        """Configurar DEAP para maximización o minimización"""
        if objective in ['sharpe', 'profit_factor', 'total_return']:
            # Maximizar
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
        else:
            # Minimizar (para drawdown, etc.)
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
    
    def _register_functions(self, toolbox, strategy, data, config):
        """Registrar funciones en DEAP toolbox"""
        # Función de evaluación
        toolbox.register("evaluate", self._evaluate_individual, 
                        strategy=strategy, data=data, config=config)
        
        # Función para crear individuo
        param_count = len(config.parameter_ranges)
        toolbox.register("attr_float", random.uniform, 0, 1)
        toolbox.register("individual", tools.initRepeat, 
                        creator.Individual, toolbox.attr_float, param_count)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Operadores genéticos
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
        
        # Paralelización
        if config.n_workers == -1:
            config.n_workers = mp.cpu_count()
        
        if config.n_workers > 1:
            pool = mp.Pool(config.n_workers)
            toolbox.register("map", pool.map)
    
    def _evaluate_individual(self, individual: list, strategy: Any, 
                           data: pd.DataFrame, config: OptimizationConfig) -> Tuple[float]:
        """Evaluar un individuo (conjunto de parámetros)"""
        try:
            # Decodificar parámetros
            params = self._decode_individual(individual, config.parameter_ranges)
            
            # Actualizar estrategia con nuevos parámetros
            strategy.config.parameters.update(params)
            
            # Ejecutar backtest
            result = self.backtest_engine.run_backtest(
                data=data,
                strategy=strategy,
                symbol='OPTIMIZE',
                commission=0.001,
                slippage_model="dynamic"
            )
            
            # Calcular fitness según objetivo
            fitness = self._calculate_fitness(result, config.objective)
            
            return (fitness,)
            
        except Exception as e:
            logger.warning(f"Error evaluando individuo: {e}")
            return (-np.inf,) if config.objective in ['sharpe', 'profit_factor', 'total_return'] else (np.inf,)
    
    def _decode_individual(self, individual: list, 
                          parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Decodificar individuo a parámetros reales"""
        params = {}
        param_names = list(parameter_ranges.keys())
        
        for i, param_name in enumerate(param_names):
            min_val, max_val = parameter_ranges[param_name]
            
            # Escalar de [0, 1] a [min_val, max_val]
            if param_name.endswith('_int'):
                # Parámetro entero
                value = int(min_val + individual[i] * (max_val - min_val))
                params[param_name.replace('_int', '')] = value
            else:
                # Parámetro float
                value = min_val + individual[i] * (max_val - min_val)
                params[param_name] = round(value, 4)
        
        return params
    
    def _calculate_fitness(self, backtest_result: Any, objective: str) -> float:
        """Calcular valor de fitness según objetivo"""
        if objective == 'sharpe':
            return backtest_result.sharpe_ratio or -10
        elif objective == 'profit_factor':
            return backtest_result.profit_factor or -10
        elif objective == 'total_return':
            return backtest_result.total_return
        elif objective == 'win_rate':
            return backtest_result.win_rate
        elif objective == 'calmar':
            if backtest_result.max_drawdown == 0:
                return -10
            return backtest_result.total_return / abs(backtest_result.max_drawdown)
        else:
            return backtest_result.sharpe_ratio or -10

class BayesianOptimizer:
    """Optimización bayesiana para parámetros de estrategias"""
    
    def __init__(self, backtest_engine):
        self.backtest_engine = backtest_engine
        
    def optimize(self, strategy, data: pd.DataFrame, 
                parameter_ranges: Dict[str, Tuple[float, float]],
                n_iter: int = 100, init_points: int = 10) -> Dict[str, Any]:
        """Optimizar usando Bayesian Optimization"""
        from bayes_opt import BayesianOptimization
        from bayes_opt.util import UtilityFunction
        
        def black_box_function(**params):
            """Función a optimizar"""
            try:
                # Actualizar estrategia
                strategy.config.parameters.update(params)
                
                # Ejecutar backtest
                result = self.backtest_engine.run_backtest(
                    data=data, strategy=strategy, symbol='OPTIMIZE'
                )
                
                # Retornar Sharpe ratio (o otra métrica)
                return result.sharpe_ratio or -10
                
            except Exception as e:
                logger.warning(f"Error en evaluación bayesiana: {e}")
                return -10
        
        # Crear optimizador
        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=parameter_ranges,
            random_state=42,
            verbose=2
        )
        
        # Optimizar
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )
        
        return {
            'best_parameters': optimizer.max['params'],
            'best_fitness': optimizer.max['target'],
            'optimization_history': optimizer.res
        }

class MultiObjectiveOptimizer:
    """Optimización multi-objetivo para trading"""
    
    def __init__(self, backtest_engine):
        self.backtest_engine = backtest_engine
        
    def optimize(self, strategy, data: pd.DataFrame,
                parameter_ranges: Dict[str, Tuple[float, float]],
                objectives: List[str] = ['sharpe', 'max_drawdown'],
                population_size: int = 100,
                generations: int = 50) -> List[Dict[str, Any]]:
        """Optimización multi-objetivo usando NSGA-II"""
        
        # Configurar DEAP para multi-objetivo
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Maximizar Sharpe, minimizar Drawdown
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        toolbox = base.Toolbox()
        
        # Registrar funciones
        param_count = len(parameter_ranges)
        toolbox.register("attr_float", random.uniform, 0, 1)
        toolbox.register("individual", tools.initRepeat, 
                        creator.Individual, toolbox.attr_float, param_count)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("evaluate", self._evaluate_multi_objective,
                        strategy=strategy, data=data, 
                        parameter_ranges=parameter_ranges,
                        objectives=objectives)
        
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                        low=[0]*param_count, up=[1]*param_count, eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded,
                        low=[0]*param_count, up=[1]*param_count, eta=20.0, indpb=1.0/param_count)
        toolbox.register("select", tools.selNSGA2)
        
        # Crear población
        population = toolbox.population(n=population_size)
        
        # Ejecutar algoritmo
        algorithms.eaMuPlusLambda(
            population, toolbox,
            mu=population_size,
            lambda_=population_size,
            cxpb=0.7,
            mutpb=0.3,
            ngen=generations,
            verbose=True
        )
        
        # Obtener frente de Pareto
        pareto_front = tools.sortNondominated(population, len(population))[0]
        
        # Convertir a formato legible
        results = []
        for ind in pareto_front:
            params = self._decode_individual(ind, parameter_ranges)
            results.append({
                'parameters': params,
                'fitness': ind.fitness.values,
                'objectives': objectives
            })
        
        return results
    
    def _evaluate_multi_objective(self, individual, strategy, data, 
                                parameter_ranges, objectives):
        """Evaluar para múltiples objetivos"""
        try:
            params = self._decode_individual(individual, parameter_ranges)
            strategy.config.parameters.update(params)
            
            result = self.backtest_engine.run_backtest(data, strategy, 'OPTIMIZE')
            
            fitness_values = []
            for objective in objectives:
                if objective == 'sharpe':
                    fitness_values.append(result.sharpe_ratio or -10)
                elif objective == 'max_drawdown':
                    fitness_values.append(-result.max_drawdown)  # Negativo porque queremos minimizar
                elif objective == 'total_return':
                    fitness_values.append(result.total_return)
                elif objective == 'profit_factor':
                    fitness_values.append(result.profit_factor or -10)
            
            return tuple(fitness_values)
            
        except Exception as e:
            logger.warning(f"Error en evaluación multi-objetivo: {e}")
            return tuple([-10] * len(objectives))
    
    def _decode_individual(self, individual, parameter_ranges):
        """Decodificar individuo (mismo método que GeneticOptimizer)"""
        params = {}
        param_names = list(parameter_ranges.keys())
        
        for i, param_name in enumerate(param_names):
            min_val, max_val = parameter_ranges[param_name]
            
            if param_name.endswith('_int'):
                value = int(min_val + individual[i] * (max_val - min_val))
                params[param_name.replace('_int', '')] = value
            else:
                value = min_val + individual[i] * (max_val - min_val)
                params[param_name] = round(value, 4)
        
        return params