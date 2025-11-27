# strategies/__init__.py
"""
Strategies Module.

Provides strategy generation, optimization, and export capabilities.
"""

from .auto_generator import (
    AutoStrategyGenerator,
    GeneratedStrategy,
    IndicatorLibrary,
    IndicatorConfig,
    IndicatorType,
    IndicatorCalculator,
    Condition,
    ConditionOperator,
    TradingRule,
    SignalType
)

from .genetic_strategy import (
    GeneticStrategyOptimizer,
    GeneticConfig,
    FitnessResult,
    StrategyChromosome,
    MultiObjectiveGeneticOptimizer
)

from .mql5_exporter import (
    MQL5Exporter,
    export_strategy_to_mql5
)

from .extended_indicators import (
    ExtendedIndicatorCalculator as ExtendedCalc,
    get_extended_indicator_configs
)

from .indicator_integration import (
    ExtendedIndicatorLibrary,
    ExtendedIndicatorCalculator,
    create_extended_generator,
    get_indicator_stats
)

from .session_ict_indicators import (
    SessionIndicators,
    OpeningRangeBreakout,
    ICTIndicators,
    SmartMoneyIndicators,
    AdditionalIndicators
)

__all__ = [
    # Core Strategy Components
    'AutoStrategyGenerator',
    'GeneratedStrategy',
    'IndicatorLibrary',
    'IndicatorConfig',
    'IndicatorType',
    'IndicatorCalculator',
    'Condition',
    'ConditionOperator',
    'TradingRule',
    'SignalType',
    
    # Extended Indicators
    'ExtendedIndicatorLibrary',
    'ExtendedIndicatorCalculator',
    'create_extended_generator',
    'get_indicator_stats',
    'get_extended_indicator_configs',
    
    # Genetic Optimization
    'GeneticStrategyOptimizer',
    'GeneticConfig',
    'FitnessResult',
    'StrategyChromosome',
    'MultiObjectiveGeneticOptimizer',
    
    # MQL5 Export
    'MQL5Exporter',
    'export_strategy_to_mql5',
    
    # Session & ICT Indicators
    'SessionIndicators',
    'OpeningRangeBreakout',
    'ICTIndicators',
    'SmartMoneyIndicators',
    'AdditionalIndicators'
]
