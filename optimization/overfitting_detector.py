# optimization/overfitting_detector.py
"""
Overfitting Detection Module.

Provides tools to detect and prevent overfitting in trading strategies.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class OverfitRisk(Enum):
    """Overfitting risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OverfitIndicator:
    """Single overfitting indicator result."""
    name: str
    value: float
    threshold: float
    is_warning: bool
    description: str
    recommendation: str


@dataclass
class OverfitReport:
    """Complete overfitting analysis report."""
    strategy_name: str
    overall_risk: OverfitRisk
    risk_score: float  # 0-100, higher = more overfit risk
    indicators: List[OverfitIndicator]
    summary: str
    recommendations: List[str]
    is_likely_overfit: bool


class OverfittingDetector:
    """
    Detects potential overfitting in trading strategies.
    
    Uses multiple indicators:
    1. Parameter Count vs Data Points ratio
    2. In-sample vs Out-of-sample performance degradation
    3. Complexity metrics
    4. Consistency across time periods
    5. Sensitivity analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Thresholds
        self.param_ratio_threshold = self.config.get('param_ratio_threshold', 50)
        self.degradation_threshold = self.config.get('degradation_threshold', 0.5)
        self.sharpe_threshold = self.config.get('sharpe_threshold', 3.0)
        self.win_rate_threshold = self.config.get('win_rate_threshold', 0.75)
        self.trade_threshold = self.config.get('min_trades', 30)
    
    def analyze(self, strategy, 
                in_sample_result,
                out_of_sample_result,
                data_points: int,
                walk_forward_results: List = None) -> OverfitReport:
        """
        Perform complete overfitting analysis.
        
        Args:
            strategy: The strategy being analyzed
            in_sample_result: Backtest result on training data
            out_of_sample_result: Backtest result on test data
            data_points: Number of data points used
            walk_forward_results: Optional walk-forward results
        
        Returns:
            OverfitReport with analysis results
        """
        indicators = []
        
        # 1. Parameter complexity check
        indicators.append(self._check_parameter_complexity(strategy, data_points))
        
        # 2. Performance degradation check
        indicators.append(self._check_performance_degradation(
            in_sample_result, out_of_sample_result
        ))
        
        # 3. Suspiciously good metrics
        indicators.append(self._check_suspicious_metrics(in_sample_result))
        
        # 4. Trade count check
        indicators.append(self._check_trade_count(in_sample_result, data_points))
        
        # 5. Consistency check (if walk-forward results available)
        if walk_forward_results:
            indicators.append(self._check_consistency(walk_forward_results))
        
        # 6. Sharpe ratio sanity check
        indicators.append(self._check_sharpe_ratio(in_sample_result))
        
        # 7. Win rate check
        indicators.append(self._check_win_rate(in_sample_result))
        
        # 8. Drawdown profile check
        indicators.append(self._check_drawdown_profile(in_sample_result))
        
        # Calculate overall risk
        warning_count = sum(1 for i in indicators if i.is_warning)
        total_indicators = len(indicators)
        
        risk_score = (warning_count / total_indicators) * 100 if total_indicators > 0 else 0
        
        # Adjust risk score based on severity
        for indicator in indicators:
            if indicator.is_warning:
                if "Critical" in indicator.description:
                    risk_score += 10
                elif "High" in indicator.description:
                    risk_score += 5
        
        risk_score = min(100, risk_score)
        
        # Determine risk level
        if risk_score < 25:
            overall_risk = OverfitRisk.LOW
        elif risk_score < 50:
            overall_risk = OverfitRisk.MEDIUM
        elif risk_score < 75:
            overall_risk = OverfitRisk.HIGH
        else:
            overall_risk = OverfitRisk.CRITICAL
        
        # Generate summary and recommendations
        summary = self._generate_summary(indicators, risk_score, overall_risk)
        recommendations = self._generate_recommendations(indicators, overall_risk)
        
        strategy_name = strategy.name if hasattr(strategy, 'name') else "Strategy"
        
        return OverfitReport(
            strategy_name=strategy_name,
            overall_risk=overall_risk,
            risk_score=risk_score,
            indicators=indicators,
            summary=summary,
            recommendations=recommendations,
            is_likely_overfit=risk_score > 50
        )
    
    def _check_parameter_complexity(self, strategy, data_points: int) -> OverfitIndicator:
        """Check if strategy has too many parameters for data size."""
        # Count parameters
        param_count = 0
        
        if hasattr(strategy, 'indicators'):
            for ind in strategy.indicators:
                param_count += len(ind.parameters)
        
        if hasattr(strategy, 'entry_rules'):
            param_count += len(strategy.entry_rules) * 2  # Approximate
        
        if hasattr(strategy, 'risk_management'):
            param_count += len(strategy.risk_management)
        
        # Simple heuristic for config objects
        if hasattr(strategy, 'config') and hasattr(strategy.config, 'parameters'):
            param_count = len(strategy.config.parameters)
        
        # Calculate ratio
        ratio = data_points / max(1, param_count)
        
        # Rule of thumb: need at least 50 data points per parameter
        is_warning = ratio < self.param_ratio_threshold
        
        description = f"Data points per parameter: {ratio:.0f}"
        if is_warning:
            description += " (Critical - too few data points)"
        
        return OverfitIndicator(
            name="Parameter Complexity",
            value=ratio,
            threshold=self.param_ratio_threshold,
            is_warning=is_warning,
            description=description,
            recommendation="Reduce number of parameters or use more data" if is_warning else "OK"
        )
    
    def _check_performance_degradation(self, is_result, oos_result) -> OverfitIndicator:
        """Check degradation between in-sample and out-of-sample."""
        is_return = is_result.total_return if is_result else 0
        oos_return = oos_result.total_return if oos_result else 0
        
        if is_return == 0:
            degradation = 0 if oos_return >= 0 else 1
        else:
            degradation = 1 - (oos_return / abs(is_return))
        
        # Clamp to reasonable range
        degradation = max(-1, min(2, degradation))
        
        is_warning = degradation > self.degradation_threshold
        
        description = f"Performance degradation: {degradation*100:.1f}%"
        if is_warning:
            description += " (High - significant IS/OOS gap)"
        
        return OverfitIndicator(
            name="Performance Degradation",
            value=degradation,
            threshold=self.degradation_threshold,
            is_warning=is_warning,
            description=description,
            recommendation="Strategy may be curve-fitted to historical data" if is_warning else "OK"
        )
    
    def _check_suspicious_metrics(self, is_result) -> OverfitIndicator:
        """Check for suspiciously perfect metrics."""
        suspicion_score = 0
        reasons = []
        
        if is_result:
            # Check win rate
            if is_result.win_rate > 85:
                suspicion_score += 30
                reasons.append(f"Very high win rate ({is_result.win_rate:.1f}%)")
            
            # Check profit factor
            if is_result.profit_factor and is_result.profit_factor > 5:
                suspicion_score += 25
                reasons.append(f"Very high profit factor ({is_result.profit_factor:.2f})")
            
            # Check max drawdown
            if is_result.max_drawdown < 2:
                suspicion_score += 20
                reasons.append(f"Very low drawdown ({is_result.max_drawdown:.2f}%)")
            
            # Check return
            if is_result.total_return > 200:
                suspicion_score += 25
                reasons.append(f"Very high return ({is_result.total_return:.1f}%)")
        
        is_warning = suspicion_score > 30
        
        description = f"Suspicion score: {suspicion_score}"
        if reasons:
            description += f" ({', '.join(reasons)})"
        
        return OverfitIndicator(
            name="Suspicious Metrics",
            value=suspicion_score,
            threshold=30,
            is_warning=is_warning,
            description=description,
            recommendation="Results may be too good to be true - verify with more tests" if is_warning else "OK"
        )
    
    def _check_trade_count(self, is_result, data_points: int) -> OverfitIndicator:
        """Check if there are enough trades for statistical significance."""
        trade_count = is_result.total_trades if is_result else 0
        
        # Calculate expected trades per data point
        trades_per_datapoint = trade_count / max(1, data_points)
        
        is_warning = trade_count < self.trade_threshold
        
        description = f"Total trades: {trade_count}"
        if is_warning:
            description += f" (Low - minimum {self.trade_threshold} recommended)"
        
        return OverfitIndicator(
            name="Trade Count",
            value=trade_count,
            threshold=self.trade_threshold,
            is_warning=is_warning,
            description=description,
            recommendation="Need more trades for statistical significance" if is_warning else "OK"
        )
    
    def _check_consistency(self, walk_forward_results: List) -> OverfitIndicator:
        """Check consistency across walk-forward windows."""
        if not walk_forward_results:
            return OverfitIndicator(
                name="Consistency",
                value=0,
                threshold=0.6,
                is_warning=True,
                description="No walk-forward results available",
                recommendation="Run walk-forward analysis"
            )
        
        # Calculate consistency metrics
        oos_returns = [r.out_of_sample_return for r in walk_forward_results 
                      if hasattr(r, 'out_of_sample_return')]
        
        if not oos_returns:
            return OverfitIndicator(
                name="Consistency",
                value=0,
                threshold=0.6,
                is_warning=True,
                description="No OOS results available",
                recommendation="Run walk-forward analysis"
            )
        
        positive_count = sum(1 for r in oos_returns if r > 0)
        consistency = positive_count / len(oos_returns)
        
        is_warning = consistency < 0.6
        
        description = f"Consistency score: {consistency*100:.1f}% ({positive_count}/{len(oos_returns)} positive windows)"
        
        return OverfitIndicator(
            name="Consistency",
            value=consistency,
            threshold=0.6,
            is_warning=is_warning,
            description=description,
            recommendation="Strategy performance is inconsistent across time periods" if is_warning else "OK"
        )
    
    def _check_sharpe_ratio(self, is_result) -> OverfitIndicator:
        """Check if Sharpe ratio is suspiciously high."""
        sharpe = is_result.sharpe_ratio if is_result and is_result.sharpe_ratio else 0
        
        # Sharpe > 3 is extremely rare in practice
        is_warning = sharpe > self.sharpe_threshold
        
        description = f"In-sample Sharpe: {sharpe:.2f}"
        if is_warning:
            description += f" (High - Sharpe > {self.sharpe_threshold} is very rare)"
        
        return OverfitIndicator(
            name="Sharpe Ratio",
            value=sharpe,
            threshold=self.sharpe_threshold,
            is_warning=is_warning,
            description=description,
            recommendation="Very high Sharpe often indicates overfitting" if is_warning else "OK"
        )
    
    def _check_win_rate(self, is_result) -> OverfitIndicator:
        """Check if win rate is suspiciously high."""
        win_rate = (is_result.win_rate / 100) if is_result else 0
        
        is_warning = win_rate > self.win_rate_threshold
        
        description = f"Win rate: {win_rate*100:.1f}%"
        if is_warning:
            description += f" (High - win rates > {self.win_rate_threshold*100}% are rare)"
        
        return OverfitIndicator(
            name="Win Rate",
            value=win_rate,
            threshold=self.win_rate_threshold,
            is_warning=is_warning,
            description=description,
            recommendation="Very high win rates may not persist in live trading" if is_warning else "OK"
        )
    
    def _check_drawdown_profile(self, is_result) -> OverfitIndicator:
        """Check drawdown profile for suspicious patterns."""
        if not is_result or not hasattr(is_result, 'equity_curve'):
            return OverfitIndicator(
                name="Drawdown Profile",
                value=0,
                threshold=0,
                is_warning=False,
                description="No equity curve available",
                recommendation="N/A"
            )
        
        max_dd = is_result.max_drawdown if is_result else 0
        
        # Check for suspiciously smooth equity curve
        is_warning = max_dd < 3  # Very low drawdown is suspicious
        
        description = f"Maximum drawdown: {max_dd:.2f}%"
        if is_warning:
            description += " (Very low - may indicate curve fitting)"
        
        return OverfitIndicator(
            name="Drawdown Profile",
            value=max_dd,
            threshold=3,
            is_warning=is_warning,
            description=description,
            recommendation="Very low drawdowns rarely persist in live trading" if is_warning else "OK"
        )
    
    def _generate_summary(self, indicators: List[OverfitIndicator],
                          risk_score: float, risk_level: OverfitRisk) -> str:
        """Generate summary text."""
        warning_count = sum(1 for i in indicators if i.is_warning)
        
        summary = f"Overfitting Risk: {risk_level.value.upper()} (Score: {risk_score:.1f}/100)\n"
        summary += f"Warnings triggered: {warning_count}/{len(indicators)}\n\n"
        
        if risk_level == OverfitRisk.CRITICAL:
            summary += "⚠️ CRITICAL: Strategy shows strong signs of overfitting. "
            summary += "Live trading is NOT recommended without major revisions."
        elif risk_level == OverfitRisk.HIGH:
            summary += "⚠️ HIGH RISK: Multiple overfitting indicators detected. "
            summary += "Further validation strongly recommended before live trading."
        elif risk_level == OverfitRisk.MEDIUM:
            summary += "⚠️ MEDIUM RISK: Some overfitting indicators present. "
            summary += "Consider additional walk-forward testing."
        else:
            summary += "✅ LOW RISK: Strategy appears robust. "
            summary += "Continue with paper trading before going live."
        
        return summary
    
    def _generate_recommendations(self, indicators: List[OverfitIndicator],
                                   risk_level: OverfitRisk) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        for indicator in indicators:
            if indicator.is_warning and indicator.recommendation != "OK":
                recommendations.append(f"• {indicator.name}: {indicator.recommendation}")
        
        # General recommendations based on risk level
        if risk_level in [OverfitRisk.HIGH, OverfitRisk.CRITICAL]:
            recommendations.extend([
                "• Run walk-forward analysis with more windows",
                "• Reduce strategy complexity (fewer indicators/rules)",
                "• Use longer out-of-sample test period",
                "• Consider Monte Carlo simulation for validation"
            ])
        elif risk_level == OverfitRisk.MEDIUM:
            recommendations.extend([
                "• Validate with additional out-of-sample data",
                "• Test on different market conditions",
                "• Start with paper trading before live"
            ])
        else:
            recommendations.append("• Strategy appears ready for paper trading")
        
        return recommendations


class SensitivityAnalyzer:
    """
    Analyzes strategy sensitivity to parameter changes.
    
    Helps identify if strategy is too sensitive (overfit)
    or robust to small parameter variations.
    """
    
    def __init__(self, variation_percent: float = 0.1):
        self.variation_percent = variation_percent
    
    def analyze(self, strategy, data: pd.DataFrame,
                backtest_engine, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sensitivity to parameter changes.
        
        Returns dict with sensitivity metrics for each parameter.
        """
        results = {}
        
        base_result = backtest_engine.run_backtest(
            data=data, strategy=strategy, symbol="SYMBOL"
        )
        base_return = base_result.total_return
        
        for param_name, param_value in parameters.items():
            if not isinstance(param_value, (int, float)):
                continue
            
            variations = []
            returns = []
            
            # Test variations
            for mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
                new_value = param_value * mult
                if isinstance(param_value, int):
                    new_value = int(new_value)
                
                # Create new strategy with varied parameter
                try:
                    varied_strategy = self._create_varied_strategy(
                        strategy, param_name, new_value
                    )
                    result = backtest_engine.run_backtest(
                        data=data, strategy=varied_strategy, symbol="SYMBOL"
                    )
                    variations.append(mult)
                    returns.append(result.total_return)
                except Exception as e:
                    logger.warning(f"Could not test variation for {param_name}: {e}")
            
            if returns:
                # Calculate sensitivity metrics
                return_std = np.std(returns)
                return_range = max(returns) - min(returns)
                sensitivity = return_std / abs(base_return) if base_return != 0 else float('inf')
                
                results[param_name] = {
                    'base_value': param_value,
                    'variations': variations,
                    'returns': returns,
                    'return_std': return_std,
                    'return_range': return_range,
                    'sensitivity': sensitivity,
                    'is_sensitive': sensitivity > 0.5  # More than 50% variation
                }
        
        # Overall sensitivity score
        sensitivities = [r['sensitivity'] for r in results.values() if r['sensitivity'] != float('inf')]
        overall_sensitivity = np.mean(sensitivities) if sensitivities else 0
        
        results['_overall'] = {
            'sensitivity': overall_sensitivity,
            'is_sensitive': overall_sensitivity > 0.3,
            'most_sensitive': max(results.keys(), key=lambda k: results[k].get('sensitivity', 0) 
                                 if k != '_overall' else 0) if results else None
        }
        
        return results
    
    def _create_varied_strategy(self, strategy, param_name: str, new_value):
        """Create a copy of strategy with varied parameter."""
        import copy
        new_strategy = copy.deepcopy(strategy)
        
        if hasattr(new_strategy, 'config') and hasattr(new_strategy.config, 'parameters'):
            new_strategy.config.parameters[param_name] = new_value
        
        return new_strategy
