# optimization/genetic_optimizer.py
"""
Genetic optimization for trading strategies.

Provides genetic algorithms, Bayesian optimization, and multi-objective optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import random
import multiprocessing as mp

logger = logging.getLogger(__name__)

# Try to import DEAP
DEAP_AVAILABLE = False
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    logger.info("DEAP not available, using basic optimization")

# Try to import bayesian optimization
BAYESIAN_AVAILABLE = False
try:
    from bayes_opt import BayesianOptimization
    BAYESIAN_AVAILABLE = True
except ImportError:
    logger.info("Bayesian optimization not available")


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    strategy_name: str
    parameter_ranges: Dict[str, Tuple[float, float]]
    objective: str = 'sharpe'  # 'sharpe', 'profit_factor', 'total_return', 'custom'
    population_size: int = 100
    generations: int = 50
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2
    tournament_size: int = 3
    hall_of_fame_size: int = 10
    n_workers: int = -1


class GeneticOptimizer:
    """Genetic optimizer for trading strategies."""
    
    def __init__(self, backtest_engine):
        self.backtest_engine = backtest_engine
        self.hall_of_fame = []
        self.optimization_history = []
    
    def optimize_strategy(
        self, 
        strategy, 
        data: pd.DataFrame,
        config: OptimizationConfig
    ) -> Dict[str, Any]:
        """Optimize strategy using genetic algorithm."""
        logger.info(f"Starting genetic optimization for {config.strategy_name}")
        
        if not DEAP_AVAILABLE:
            return self._basic_optimization(strategy, data, config)
        
        try:
            return self._deap_optimization(strategy, data, config)
        except Exception as e:
            logger.warning(f"DEAP optimization failed: {e}, falling back to basic")
            return self._basic_optimization(strategy, data, config)
    
    def _deap_optimization(
        self, 
        strategy, 
        data: pd.DataFrame,
        config: OptimizationConfig
    ) -> Dict[str, Any]:
        """Optimization using DEAP library."""
        # Setup DEAP
        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        if hasattr(creator, 'Individual'):
            del creator.Individual
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Register functions
        param_count = len(config.parameter_ranges)
        toolbox.register("attr_float", random.uniform, 0, 1)
        toolbox.register("individual", tools.initRepeat, 
                        creator.Individual, toolbox.attr_float, param_count)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Evaluation function
        def evaluate(individual):
            return self._evaluate_individual(individual, strategy, data, config)
        
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
        
        # Create population
        population = toolbox.population(n=config.population_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Hall of Fame
        hof = tools.HallOfFame(config.hall_of_fame_size)
        
        # Run genetic algorithm
        population, logbook = algorithms.eaSimple(
            population, toolbox,
            cxpb=config.crossover_prob,
            mutpb=config.mutation_prob,
            ngen=config.generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )
        
        # Get best results
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
            'optimization_history': [dict(gen) for gen in logbook],
            'generations': config.generations,
            'population_size': config.population_size
        }
        
        logger.info(f"Optimization completed. Best fitness: {best_fitness:.4f}")
        return result
    
    def _basic_optimization(
        self, 
        strategy, 
        data: pd.DataFrame,
        config: OptimizationConfig
    ) -> Dict[str, Any]:
        """Basic random search optimization when DEAP is not available."""
        logger.info("Using basic random search optimization")
        
        best_params = None
        best_fitness = -np.inf
        
        n_iterations = config.population_size * config.generations
        
        for i in range(n_iterations):
            # Generate random individual
            individual = [random.uniform(0, 1) for _ in range(len(config.parameter_ranges))]
            fitness = self._evaluate_individual(individual, strategy, data, config)[0]
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = self._decode_individual(individual, config.parameter_ranges)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Iteration {i+1}/{n_iterations}, best fitness: {best_fitness:.4f}")
        
        return {
            'best_parameters': best_params,
            'best_fitness': best_fitness,
            'hall_of_fame': [{'parameters': best_params, 'fitness': best_fitness}],
            'optimization_history': [],
            'generations': config.generations,
            'population_size': config.population_size
        }
    
    def _evaluate_individual(
        self, 
        individual: list, 
        strategy: Any,
        data: pd.DataFrame, 
        config: OptimizationConfig
    ) -> Tuple[float]:
        """Evaluate an individual (parameter set)."""
        try:
            # Decode parameters
            params = self._decode_individual(individual, config.parameter_ranges)
            
            # Update strategy with new parameters
            strategy.config.parameters.update(params)
            
            # Run backtest
            result = self.backtest_engine.run_backtest(
                data=data,
                strategy=strategy,
                symbol='OPTIMIZE',
                commission=0.001,
                slippage_model="dynamic"
            )
            
            # Calculate fitness
            fitness = self._calculate_fitness(result, config.objective)
            
            return (fitness,)
            
        except Exception as e:
            logger.warning(f"Error evaluating individual: {e}")
            return (-np.inf,)
    
    def _decode_individual(
        self, 
        individual: list,
        parameter_ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Decode individual to actual parameters."""
        params = {}
        param_names = list(parameter_ranges.keys())
        
        for i, param_name in enumerate(param_names):
            if i >= len(individual):
                break
            
            min_val, max_val = parameter_ranges[param_name]
            
            # Scale from [0, 1] to [min_val, max_val]
            if param_name.endswith('_int'):
                # Integer parameter
                value = int(min_val + individual[i] * (max_val - min_val))
                params[param_name.replace('_int', '')] = value
            else:
                # Float parameter
                value = min_val + individual[i] * (max_val - min_val)
                params[param_name] = round(value, 4)
        
        return params
    
    def _calculate_fitness(self, backtest_result: Any, objective: str) -> float:
        """Calculate fitness value based on objective."""
        if objective == 'sharpe':
            return backtest_result.sharpe_ratio or -10
        elif objective == 'profit_factor':
            pf = backtest_result.profit_factor
            return pf if pf and pf != float('inf') else -10
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
    """Bayesian optimization for strategy parameters."""
    
    def __init__(self, backtest_engine):
        self.backtest_engine = backtest_engine
    
    def optimize(
        self, 
        strategy, 
        data: pd.DataFrame,
        parameter_ranges: Dict[str, Tuple[float, float]],
        n_iter: int = 100, 
        init_points: int = 10
    ) -> Dict[str, Any]:
        """Optimize using Bayesian Optimization."""
        if not BAYESIAN_AVAILABLE:
            logger.warning("Bayesian optimization not available")
            return {'best_parameters': {}, 'best_fitness': -np.inf}
        
        def black_box_function(**params):
            """Function to optimize."""
            try:
                # Convert float params to int where needed
                converted_params = {}
                for key, value in params.items():
                    if key.endswith('_int'):
                        converted_params[key.replace('_int', '')] = int(value)
                    else:
                        converted_params[key] = value
                
                strategy.config.parameters.update(converted_params)
                
                result = self.backtest_engine.run_backtest(
                    data=data, strategy=strategy, symbol='OPTIMIZE'
                )
                
                return result.sharpe_ratio or -10
                
            except Exception as e:
                logger.warning(f"Error in Bayesian evaluation: {e}")
                return -10
        
        try:
            optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds=parameter_ranges,
                random_state=42,
                verbose=2
            )
            
            optimizer.maximize(
                init_points=init_points,
                n_iter=n_iter,
            )
            
            return {
                'best_parameters': optimizer.max['params'],
                'best_fitness': optimizer.max['target'],
                'optimization_history': optimizer.res
            }
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            return {'best_parameters': {}, 'best_fitness': -np.inf}


class MultiObjectiveOptimizer:
    """Multi-objective optimization for trading strategies."""
    
    def __init__(self, backtest_engine):
        self.backtest_engine = backtest_engine
    
    def optimize(
        self, 
        strategy, 
        data: pd.DataFrame,
        parameter_ranges: Dict[str, Tuple[float, float]],
        objectives: List[str] = None,
        population_size: int = 100,
        generations: int = 50
    ) -> List[Dict[str, Any]]:
        """Multi-objective optimization using NSGA-II."""
        if objectives is None:
            objectives = ['sharpe', 'max_drawdown']
        
        if not DEAP_AVAILABLE:
            logger.warning("DEAP not available for multi-objective optimization")
            return []
        
        try:
            # Setup DEAP for multi-objective
            if hasattr(creator, 'FitnessMulti'):
                del creator.FitnessMulti
            if hasattr(creator, 'IndividualMulti'):
                del creator.IndividualMulti
            
            # Weights: maximize Sharpe, minimize drawdown
            weights = tuple(1.0 if obj != 'max_drawdown' else -1.0 for obj in objectives)
            creator.create("FitnessMulti", base.Fitness, weights=weights)
            creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)
            
            toolbox = base.Toolbox()
            
            param_count = len(parameter_ranges)
            toolbox.register("attr_float", random.uniform, 0, 1)
            toolbox.register("individual", tools.initRepeat, 
                            creator.IndividualMulti, toolbox.attr_float, param_count)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            def evaluate_multi(individual):
                return self._evaluate_multi_objective(
                    individual, strategy, data, parameter_ranges, objectives
                )
            
            toolbox.register("evaluate", evaluate_multi)
            toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                            low=[0]*param_count, up=[1]*param_count, eta=20.0)
            toolbox.register("mutate", tools.mutPolynomialBounded,
                            low=[0]*param_count, up=[1]*param_count, eta=20.0, indpb=1.0/param_count)
            toolbox.register("select", tools.selNSGA2)
            
            # Create population
            population = toolbox.population(n=population_size)
            
            # Evaluate initial population
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            # Run NSGA-II
            for gen in range(generations):
                offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.3)
                
                fits = list(map(toolbox.evaluate, offspring))
                for ind, fit in zip(offspring, fits):
                    ind.fitness.values = fit
                
                population = toolbox.select(population + offspring, k=population_size)
                
                if (gen + 1) % 10 == 0:
                    logger.info(f"Generation {gen + 1}/{generations}")
            
            # Get Pareto front
            pareto_front = tools.sortNondominated(population, len(population))[0]
            
            results = []
            for ind in pareto_front:
                params = self._decode_individual(ind, parameter_ranges)
                results.append({
                    'parameters': params,
                    'fitness': ind.fitness.values,
                    'objectives': objectives
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Multi-objective optimization failed: {e}")
            return []
    
    def _evaluate_multi_objective(
        self, 
        individual, 
        strategy, 
        data,
        parameter_ranges, 
        objectives
    ) -> Tuple:
        """Evaluate for multiple objectives."""
        try:
            params = self._decode_individual(individual, parameter_ranges)
            strategy.config.parameters.update(params)
            
            result = self.backtest_engine.run_backtest(data, strategy, 'OPTIMIZE')
            
            fitness_values = []
            for objective in objectives:
                if objective == 'sharpe':
                    fitness_values.append(result.sharpe_ratio or -10)
                elif objective == 'max_drawdown':
                    fitness_values.append(-result.max_drawdown)
                elif objective == 'total_return':
                    fitness_values.append(result.total_return)
                elif objective == 'profit_factor':
                    pf = result.profit_factor
                    fitness_values.append(pf if pf and pf != float('inf') else -10)
            
            return tuple(fitness_values)
            
        except Exception as e:
            logger.warning(f"Error in multi-objective evaluation: {e}")
            return tuple([-10] * len(objectives))
    
    def _decode_individual(
        self, 
        individual, 
        parameter_ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Decode individual to parameters."""
        params = {}
        param_names = list(parameter_ranges.keys())
        
        for i, param_name in enumerate(param_names):
            if i >= len(individual):
                break
            
            min_val, max_val = parameter_ranges[param_name]
            
            if param_name.endswith('_int'):
                value = int(min_val + individual[i] * (max_val - min_val))
                params[param_name.replace('_int', '')] = value
            else:
                value = min_val + individual[i] * (max_val - min_val)
                params[param_name] = round(value, 4)
        
        return params
