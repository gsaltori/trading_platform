# strategies/genetic_strategy.py
"""
Genetic Algorithm for Strategy Evolution.

Evolves trading strategies using genetic algorithms to find optimal
combinations of indicators, parameters, and rules.
"""

import logging
import random
import copy
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from datetime import datetime
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

from .auto_generator import (
    GeneratedStrategy, AutoStrategyGenerator, IndicatorConfig,
    TradingRule, Condition, ConditionOperator, SignalType, IndicatorType
)

logger = logging.getLogger(__name__)


@dataclass
class FitnessResult:
    """Result of fitness evaluation."""
    strategy_id: str
    fitness: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_count: int = 5
    tournament_size: int = 5
    min_trades: int = 20
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        'sharpe_ratio': 0.3,
        'profit_factor': 0.25,
        'total_return': 0.2,
        'win_rate': 0.15,
        'max_drawdown': 0.1  # Negative weight - lower is better
    })
    parallel: bool = True
    n_workers: int = -1  # -1 for auto


class StrategyChromosome:
    """
    Chromosome representation of a strategy for genetic operations.
    """
    
    def __init__(self, strategy: GeneratedStrategy = None):
        self.strategy = strategy
        self.fitness: Optional[FitnessResult] = None
        self.age: int = 0
    
    def get_genes(self) -> Dict[str, Any]:
        """Extract genes from strategy."""
        if not self.strategy:
            return {}
        
        genes = {
            'indicators': [],
            'entry_rules': [],
            'exit_rules': [],
            'risk_management': self.strategy.risk_management.copy()
        }
        
        for ind in self.strategy.indicators:
            genes['indicators'].append({
                'name': ind.name,
                'type': ind.indicator_type.value,
                'parameters': ind.parameters.copy()
            })
        
        for rule in self.strategy.entry_rules:
            genes['entry_rules'].append(rule.to_dict())
        
        for rule in self.strategy.exit_rules:
            genes['exit_rules'].append(rule.to_dict())
        
        return genes
    
    def set_genes(self, genes: Dict[str, Any]):
        """Set genes to strategy."""
        # This would rebuild the strategy from genes
        pass
    
    def copy(self) -> 'StrategyChromosome':
        """Create a deep copy of the chromosome."""
        new_chromosome = StrategyChromosome()
        new_chromosome.strategy = copy.deepcopy(self.strategy)
        new_chromosome.age = self.age
        return new_chromosome


class GeneticStrategyOptimizer:
    """
    Genetic Algorithm optimizer for trading strategies.
    
    Evolves a population of strategies to find optimal configurations.
    """
    
    def __init__(self, config: GeneticConfig = None):
        self.config = config or GeneticConfig()
        self.generator = AutoStrategyGenerator()
        self.population: List[StrategyChromosome] = []
        self.best_strategy: Optional[StrategyChromosome] = None
        self.generation_history: List[Dict] = []
        
        # Callbacks
        self.on_generation_complete: Optional[Callable] = None
        self.on_new_best: Optional[Callable] = None
        
        # Parallel processing
        if self.config.n_workers == -1:
            self.n_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.n_workers = self.config.n_workers
    
    def initialize_population(self, symbols: List[str] = None,
                             timeframe: str = "H1") -> List[StrategyChromosome]:
        """Initialize random population."""
        self.population = []
        symbols = symbols or ["EURUSD"]
        
        for i in range(self.config.population_size):
            strategy = self.generator.generate_random_strategy(
                name=f"Gen0_Ind{i}",
                symbols=symbols,
                timeframe=timeframe
            )
            chromosome = StrategyChromosome(strategy)
            self.population.append(chromosome)
        
        logger.info(f"Initialized population with {len(self.population)} individuals")
        return self.population
    
    def evaluate_fitness(self, chromosome: StrategyChromosome,
                         data: pd.DataFrame,
                         backtest_engine) -> FitnessResult:
        """Evaluate fitness of a single chromosome."""
        strategy = chromosome.strategy
        
        try:
            # Calculate indicators and signals
            signals_data = self.generator.evaluate_strategy(strategy, data)
            
            # Create a temporary strategy object for backtesting
            from strategies.strategy_engine import StrategyConfig, BaseStrategy
            
            class TempStrategy(BaseStrategy):
                def __init__(self, name, signals):
                    config = StrategyConfig(
                        name=name,
                        symbols=[],
                        timeframe="H1"
                    )
                    super().__init__(config)
                    self.precalculated_signals = signals
                
                def calculate_indicators(self, data):
                    return data
                
                def generate_signals(self, data):
                    if 'signal' not in data.columns:
                        data['signal'] = self.precalculated_signals.get('signal', 0)
                    return data
            
            # Run backtest
            temp_strategy = TempStrategy(strategy.name, signals_data)
            result = backtest_engine.run_backtest(
                data=signals_data,
                strategy=temp_strategy,
                symbol=strategy.symbols[0] if strategy.symbols else "SYMBOL"
            )
            
            # Extract metrics
            total_return = result.total_return
            sharpe = result.sharpe_ratio if result.sharpe_ratio else 0
            max_dd = result.max_drawdown
            win_rate = result.win_rate
            profit_factor = result.profit_factor if result.profit_factor and result.profit_factor != float('inf') else 0
            total_trades = result.total_trades
            
            # Calculate composite fitness
            weights = self.config.fitness_weights
            
            # Normalize metrics
            norm_return = np.tanh(total_return / 100)  # Normalize to [-1, 1]
            norm_sharpe = np.tanh(sharpe / 2)
            norm_dd = 1 - (max_dd / 100)  # Lower is better
            norm_wr = win_rate / 100
            norm_pf = np.tanh((profit_factor - 1) / 2) if profit_factor > 0 else -1
            
            # Penalize strategies with too few trades
            trade_penalty = 1.0 if total_trades >= self.config.min_trades else total_trades / self.config.min_trades
            
            fitness = (
                weights.get('total_return', 0.2) * norm_return +
                weights.get('sharpe_ratio', 0.3) * norm_sharpe +
                weights.get('max_drawdown', 0.1) * norm_dd +
                weights.get('win_rate', 0.15) * norm_wr +
                weights.get('profit_factor', 0.25) * norm_pf
            ) * trade_penalty
            
            return FitnessResult(
                strategy_id=strategy.id,
                fitness=fitness,
                total_return=total_return,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                metrics={
                    'norm_return': norm_return,
                    'norm_sharpe': norm_sharpe,
                    'norm_dd': norm_dd,
                    'norm_wr': norm_wr,
                    'norm_pf': norm_pf,
                    'trade_penalty': trade_penalty
                }
            )
            
        except Exception as e:
            logger.warning(f"Error evaluating fitness for {strategy.id}: {e}")
            return FitnessResult(
                strategy_id=strategy.id,
                fitness=-1.0,
                total_return=0,
                sharpe_ratio=0,
                max_drawdown=100,
                win_rate=0,
                profit_factor=0,
                total_trades=0
            )
    
    def evaluate_population(self, data: pd.DataFrame, backtest_engine):
        """Evaluate fitness of entire population."""
        if self.config.parallel and self.n_workers > 1:
            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for chromosome in self.population:
                    future = executor.submit(
                        self.evaluate_fitness, chromosome, data, backtest_engine
                    )
                    futures.append((chromosome, future))
                
                for chromosome, future in futures:
                    try:
                        chromosome.fitness = future.result(timeout=60)
                    except Exception as e:
                        logger.warning(f"Fitness evaluation timeout: {e}")
                        chromosome.fitness = FitnessResult(
                            strategy_id=chromosome.strategy.id,
                            fitness=-1.0,
                            total_return=0, sharpe_ratio=0, max_drawdown=100,
                            win_rate=0, profit_factor=0, total_trades=0
                        )
        else:
            # Sequential evaluation
            for chromosome in self.population:
                chromosome.fitness = self.evaluate_fitness(chromosome, data, backtest_engine)
    
    def selection(self) -> List[StrategyChromosome]:
        """Select parents using tournament selection."""
        selected = []
        
        for _ in range(len(self.population)):
            # Tournament selection
            tournament = random.sample(self.population, self.config.tournament_size)
            winner = max(tournament, key=lambda c: c.fitness.fitness if c.fitness else -float('inf'))
            selected.append(winner.copy())
        
        return selected
    
    def crossover(self, parent1: StrategyChromosome,
                  parent2: StrategyChromosome) -> Tuple[StrategyChromosome, StrategyChromosome]:
        """Perform crossover between two parents."""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Crossover indicators
        if parent1.strategy.indicators and parent2.strategy.indicators:
            crossover_point = random.randint(1, min(
                len(parent1.strategy.indicators),
                len(parent2.strategy.indicators)
            ))
            
            child1.strategy.indicators = (
                parent1.strategy.indicators[:crossover_point] +
                parent2.strategy.indicators[crossover_point:]
            )
            child2.strategy.indicators = (
                parent2.strategy.indicators[:crossover_point] +
                parent1.strategy.indicators[crossover_point:]
            )
        
        # Crossover entry rules
        if parent1.strategy.entry_rules and parent2.strategy.entry_rules:
            # Swap some rules
            if random.random() > 0.5:
                rule_idx = random.randint(0, min(
                    len(parent1.strategy.entry_rules),
                    len(parent2.strategy.entry_rules)
                ) - 1)
                child1.strategy.entry_rules[rule_idx] = copy.deepcopy(
                    parent2.strategy.entry_rules[rule_idx]
                )
                child2.strategy.entry_rules[rule_idx] = copy.deepcopy(
                    parent1.strategy.entry_rules[rule_idx]
                )
        
        # Crossover risk management
        for key in child1.strategy.risk_management:
            if random.random() > 0.5:
                child1.strategy.risk_management[key], child2.strategy.risk_management[key] = \
                    child2.strategy.risk_management[key], child1.strategy.risk_management[key]
        
        # Update IDs
        child1.strategy.id = f"{child1.strategy.id[:4]}_c1"
        child2.strategy.id = f"{child2.strategy.id[:4]}_c2"
        
        return child1, child2
    
    def mutate(self, chromosome: StrategyChromosome) -> StrategyChromosome:
        """Mutate a chromosome."""
        if random.random() > self.config.mutation_rate:
            return chromosome
        
        mutated = chromosome.copy()
        strategy = mutated.strategy
        
        # Choose mutation type
        mutation_type = random.choice([
            'indicator_param',
            'indicator_add_remove',
            'rule_condition',
            'rule_logic',
            'risk_param'
        ])
        
        try:
            if mutation_type == 'indicator_param' and strategy.indicators:
                # Mutate indicator parameters
                ind_idx = random.randint(0, len(strategy.indicators) - 1)
                indicator = strategy.indicators[ind_idx]
                
                if indicator.parameters:
                    param_key = random.choice(list(indicator.parameters.keys()))
                    current_value = indicator.parameters[param_key]
                    
                    # Mutate by ±20%
                    if isinstance(current_value, float):
                        new_value = current_value * random.uniform(0.8, 1.2)
                        new_value = round(new_value, 2)
                    else:
                        delta = max(1, int(current_value * 0.2))
                        new_value = current_value + random.randint(-delta, delta)
                        new_value = max(1, new_value)
                    
                    indicator.parameters[param_key] = new_value
            
            elif mutation_type == 'indicator_add_remove' and strategy.indicators:
                if random.random() > 0.5 and len(strategy.indicators) > 2:
                    # Remove an indicator
                    idx = random.randint(0, len(strategy.indicators) - 1)
                    strategy.indicators.pop(idx)
                else:
                    # Add a new indicator
                    new_indicator = self.generator.indicator_library.get_random(1)[0]
                    params = {}
                    for param_name, param_config in new_indicator.parameters.items():
                        if isinstance(param_config, dict):
                            min_val = param_config.get('min', 5)
                            max_val = param_config.get('max', 50)
                            if isinstance(min_val, float):
                                params[param_name] = round(random.uniform(min_val, max_val), 2)
                            else:
                                params[param_name] = random.randint(min_val, max_val)
                    
                    new_config = IndicatorConfig(
                        name=new_indicator.name,
                        indicator_type=new_indicator.indicator_type,
                        parameters=params,
                        output_columns=new_indicator.output_columns
                    )
                    strategy.indicators.append(new_config)
            
            elif mutation_type == 'rule_condition' and strategy.entry_rules:
                # Mutate a condition in a rule
                rule_idx = random.randint(0, len(strategy.entry_rules) - 1)
                rule = strategy.entry_rules[rule_idx]
                
                if rule.conditions:
                    cond_idx = random.randint(0, len(rule.conditions) - 1)
                    condition = rule.conditions[cond_idx]
                    
                    # Mutate operator
                    operators = list(ConditionOperator)
                    condition.operator = random.choice(operators)
            
            elif mutation_type == 'rule_logic' and strategy.entry_rules:
                # Toggle AND/OR logic
                rule_idx = random.randint(0, len(strategy.entry_rules) - 1)
                strategy.entry_rules[rule_idx].logic = "OR" if strategy.entry_rules[rule_idx].logic == "AND" else "AND"
            
            elif mutation_type == 'risk_param':
                # Mutate risk management
                risk_keys = list(strategy.risk_management.keys())
                numeric_keys = [k for k in risk_keys 
                               if isinstance(strategy.risk_management[k], (int, float))]
                
                if numeric_keys:
                    key = random.choice(numeric_keys)
                    current = strategy.risk_management[key]
                    if isinstance(current, float):
                        strategy.risk_management[key] = round(current * random.uniform(0.8, 1.2), 3)
                    else:
                        strategy.risk_management[key] = int(current * random.uniform(0.8, 1.2))
        
        except Exception as e:
            logger.warning(f"Mutation error: {e}")
        
        mutated.strategy.id = f"{mutated.strategy.id[:4]}_m"
        return mutated
    
    def evolve_generation(self, data: pd.DataFrame, backtest_engine) -> Dict:
        """Evolve one generation."""
        # Evaluate current population
        self.evaluate_population(data, backtest_engine)
        
        # Sort by fitness
        self.population.sort(key=lambda c: c.fitness.fitness if c.fitness else -float('inf'), reverse=True)
        
        # Update best
        if self.population and self.population[0].fitness:
            if self.best_strategy is None or \
               self.population[0].fitness.fitness > (self.best_strategy.fitness.fitness if self.best_strategy.fitness else -float('inf')):
                self.best_strategy = self.population[0].copy()
                if self.on_new_best:
                    self.on_new_best(self.best_strategy)
        
        # Elitism - keep best individuals
        new_population = [c.copy() for c in self.population[:self.config.elitism_count]]
        
        # Selection
        selected = self.selection()
        
        # Crossover and mutation
        while len(new_population) < self.config.population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.config.population_size:
                new_population.append(child2)
        
        # Update ages
        for chromosome in new_population:
            chromosome.age += 1
        
        self.population = new_population
        
        # Calculate generation statistics
        fitnesses = [c.fitness.fitness for c in self.population if c.fitness]
        stats = {
            'best_fitness': max(fitnesses) if fitnesses else 0,
            'avg_fitness': np.mean(fitnesses) if fitnesses else 0,
            'min_fitness': min(fitnesses) if fitnesses else 0,
            'std_fitness': np.std(fitnesses) if fitnesses else 0,
            'best_return': self.population[0].fitness.total_return if self.population[0].fitness else 0,
            'best_sharpe': self.population[0].fitness.sharpe_ratio if self.population[0].fitness else 0,
            'best_trades': self.population[0].fitness.total_trades if self.population[0].fitness else 0
        }
        
        return stats
    
    def run(self, data: pd.DataFrame, backtest_engine,
            symbols: List[str] = None, timeframe: str = "H1",
            callback: Callable = None) -> GeneratedStrategy:
        """
        Run the complete genetic algorithm optimization.
        
        Args:
            data: Historical price data
            backtest_engine: Backtest engine instance
            symbols: List of symbols to trade
            timeframe: Trading timeframe
            callback: Optional callback(generation, stats) for progress updates
        
        Returns:
            Best strategy found
        """
        logger.info(f"Starting genetic optimization: {self.config.generations} generations, "
                   f"{self.config.population_size} population")
        
        # Initialize population
        self.initialize_population(symbols, timeframe)
        
        # Evolution loop
        for generation in range(self.config.generations):
            stats = self.evolve_generation(data, backtest_engine)
            
            self.generation_history.append({
                'generation': generation,
                **stats
            })
            
            if callback:
                callback(generation, stats)
            
            if self.on_generation_complete:
                self.on_generation_complete(generation, stats)
            
            logger.info(f"Gen {generation + 1}/{self.config.generations}: "
                       f"Best={stats['best_fitness']:.4f}, "
                       f"Avg={stats['avg_fitness']:.4f}, "
                       f"Return={stats['best_return']:.2f}%")
            
            # Early stopping if we've found a great strategy
            if stats['best_fitness'] > 0.9 and stats['best_return'] > 50:
                logger.info("Early stopping - excellent strategy found")
                break
        
        # Return best strategy
        if self.best_strategy:
            self.best_strategy.strategy.fitness_score = self.best_strategy.fitness.fitness
            return self.best_strategy.strategy
        elif self.population:
            return self.population[0].strategy
        else:
            return None
    
    def get_top_strategies(self, n: int = 10) -> List[GeneratedStrategy]:
        """Get top N strategies from current population."""
        sorted_pop = sorted(
            self.population,
            key=lambda c: c.fitness.fitness if c.fitness else -float('inf'),
            reverse=True
        )
        
        return [c.strategy for c in sorted_pop[:n]]
    
    def get_evolution_report(self) -> Dict:
        """Get a report of the evolution process."""
        return {
            'generations_run': len(self.generation_history),
            'best_fitness': self.best_strategy.fitness.fitness if self.best_strategy and self.best_strategy.fitness else 0,
            'best_strategy': self.best_strategy.strategy.to_dict() if self.best_strategy else None,
            'history': self.generation_history
        }


class MultiObjectiveGeneticOptimizer(GeneticStrategyOptimizer):
    """
    Multi-objective genetic optimizer using NSGA-II.
    
    Optimizes for multiple objectives simultaneously:
    - Return
    - Risk (Drawdown)
    - Sharpe Ratio
    - Consistency (Win Rate)
    """
    
    def __init__(self, config: GeneticConfig = None):
        super().__init__(config)
        self.pareto_front: List[StrategyChromosome] = []
    
    def dominates(self, fitness1: FitnessResult, fitness2: FitnessResult) -> bool:
        """Check if fitness1 dominates fitness2 (Pareto dominance)."""
        objectives1 = [
            fitness1.total_return,
            -fitness1.max_drawdown,  # Minimize
            fitness1.sharpe_ratio,
            fitness1.win_rate
        ]
        
        objectives2 = [
            fitness2.total_return,
            -fitness2.max_drawdown,
            fitness2.sharpe_ratio,
            fitness2.win_rate
        ]
        
        better_in_any = False
        worse_in_any = False
        
        for o1, o2 in zip(objectives1, objectives2):
            if o1 > o2:
                better_in_any = True
            elif o1 < o2:
                worse_in_any = True
        
        return better_in_any and not worse_in_any
    
    def calculate_crowding_distance(self, population: List[StrategyChromosome]) -> Dict[str, float]:
        """Calculate crowding distance for diversity preservation."""
        if len(population) <= 2:
            return {c.strategy.id: float('inf') for c in population}
        
        distances = {c.strategy.id: 0 for c in population}
        
        objectives = ['total_return', 'max_drawdown', 'sharpe_ratio', 'win_rate']
        
        for obj in objectives:
            # Sort by objective
            sorted_pop = sorted(
                population,
                key=lambda c: getattr(c.fitness, obj) if c.fitness else 0
            )
            
            # Boundary points have infinite distance
            distances[sorted_pop[0].strategy.id] = float('inf')
            distances[sorted_pop[-1].strategy.id] = float('inf')
            
            # Calculate distance for middle points
            obj_range = (
                (getattr(sorted_pop[-1].fitness, obj) if sorted_pop[-1].fitness else 0) -
                (getattr(sorted_pop[0].fitness, obj) if sorted_pop[0].fitness else 0)
            )
            
            if obj_range > 0:
                for i in range(1, len(sorted_pop) - 1):
                    prev_val = getattr(sorted_pop[i-1].fitness, obj) if sorted_pop[i-1].fitness else 0
                    next_val = getattr(sorted_pop[i+1].fitness, obj) if sorted_pop[i+1].fitness else 0
                    distances[sorted_pop[i].strategy.id] += (next_val - prev_val) / obj_range
        
        return distances
    
    def fast_non_dominated_sort(self, population: List[StrategyChromosome]) -> List[List[StrategyChromosome]]:
        """Perform fast non-dominated sorting."""
        fronts = [[]]
        domination_count = {c.strategy.id: 0 for c in population}
        dominated_solutions = {c.strategy.id: [] for c in population}
        
        for p in population:
            for q in population:
                if p.strategy.id == q.strategy.id:
                    continue
                
                if p.fitness and q.fitness:
                    if self.dominates(p.fitness, q.fitness):
                        dominated_solutions[p.strategy.id].append(q)
                    elif self.dominates(q.fitness, p.fitness):
                        domination_count[p.strategy.id] += 1
            
            if domination_count[p.strategy.id] == 0:
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p.strategy.id]:
                    domination_count[q.strategy.id] -= 1
                    if domination_count[q.strategy.id] == 0:
                        next_front.append(q)
            
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def selection(self) -> List[StrategyChromosome]:
        """NSGA-II selection using non-dominated sorting and crowding distance."""
        fronts = self.fast_non_dominated_sort(self.population)
        
        selected = []
        front_idx = 0
        
        while len(selected) < len(self.population) and front_idx < len(fronts):
            front = fronts[front_idx]
            
            if len(selected) + len(front) <= len(self.population):
                selected.extend([c.copy() for c in front])
            else:
                # Need to select some from this front based on crowding distance
                distances = self.calculate_crowding_distance(front)
                sorted_front = sorted(
                    front,
                    key=lambda c: distances[c.strategy.id],
                    reverse=True
                )
                
                remaining = len(self.population) - len(selected)
                selected.extend([c.copy() for c in sorted_front[:remaining]])
            
            front_idx += 1
        
        # Update Pareto front
        if fronts:
            self.pareto_front = [c.copy() for c in fronts[0]]
        
        return selected
    
    def get_pareto_front(self) -> List[GeneratedStrategy]:
        """Get strategies on the Pareto front."""
        return [c.strategy for c in self.pareto_front]
