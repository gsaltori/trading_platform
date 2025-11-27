# strategies/mql5_exporter.py
"""
MQL5 Code Exporter.

Exports generated strategies to MQL5 code for use in MetaTrader 5.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from .auto_generator import (
    GeneratedStrategy, IndicatorConfig, TradingRule, Condition,
    ConditionOperator, SignalType, IndicatorType
)

logger = logging.getLogger(__name__)


class MQL5Exporter:
    """
    Exports trading strategies to MQL5 Expert Advisor code.
    """
    
    def __init__(self):
        self.indent = "   "
    
    def export(self, strategy: GeneratedStrategy, 
               output_path: str = None,
               include_comments: bool = True) -> str:
        """
        Export a strategy to MQL5 code.
        
        Args:
            strategy: Strategy to export
            output_path: Optional file path to save the code
            include_comments: Include explanatory comments
        
        Returns:
            MQL5 code as string
        """
        code = self._generate_header(strategy, include_comments)
        code += self._generate_properties(strategy)
        code += self._generate_inputs(strategy)
        code += self._generate_global_variables(strategy)
        code += self._generate_indicator_handles(strategy)
        code += self._generate_oninit(strategy)
        code += self._generate_ondeinit()
        code += self._generate_ontick(strategy)
        code += self._generate_helper_functions(strategy)
        code += self._generate_indicator_functions(strategy)
        code += self._generate_signal_functions(strategy)
        code += self._generate_trading_functions(strategy)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.info(f"Exported strategy to {output_path}")
        
        return code
    
    def _generate_header(self, strategy: GeneratedStrategy, include_comments: bool) -> str:
        """Generate file header."""
        header = f"""//+------------------------------------------------------------------+
//|                                        {strategy.name}.mq5 |
//|                                 Auto-Generated Trading Strategy |
//|                            Generated: {strategy.generation_date.strftime('%Y-%m-%d %H:%M')} |
//+------------------------------------------------------------------+
"""
        
        if include_comments:
            header += f"""//+------------------------------------------------------------------+
//| Strategy Description:                                            |
//| {strategy.description[:60]}
//|                                                                  |
//| Indicators Used:                                                 |
"""
            for ind in strategy.indicators:
                header += f"//|   - {ind.name} ({ind.indicator_type.value})\n"
            
            header += f"""//|                                                                  |
//| Risk Management:                                                 |
//|   - ATR Multiplier: {strategy.risk_management.get('atr_multiplier', 2.0)}
//|   - Risk/Reward: {strategy.risk_management.get('risk_reward_ratio', 1.5)}
//|   - Max Risk per Trade: {strategy.risk_management.get('max_risk_per_trade', 0.02)*100}%
//+------------------------------------------------------------------+

"""
        return header
    
    def _generate_properties(self, strategy: GeneratedStrategy) -> str:
        """Generate EA properties."""
        return f"""#property copyright "Auto-Generated Strategy"
#property link      ""
#property version   "1.00"
#property strict

#include <Trade\\Trade.mqh>

"""
    
    def _generate_inputs(self, strategy: GeneratedStrategy) -> str:
        """Generate input parameters."""
        code = """//--- Input Parameters
input group "=== General Settings ==="
input int      MagicNumber = 123456;           // Magic Number
input double   RiskPercent = 2.0;              // Risk per trade (%)
input double   LotSize = 0.1;                  // Fixed lot size (if not using risk %)
input bool     UseFixedLot = false;            // Use fixed lot size
input ENUM_TIMEFRAMES Timeframe = PERIOD_H1;   // Timeframe

input group "=== Risk Management ==="
"""
        
        rm = strategy.risk_management
        code += f"""input double   ATRMultiplierSL = {rm.get('atr_multiplier', 2.0)};  // ATR multiplier for Stop Loss
input double   RiskRewardRatio = {rm.get('risk_reward_ratio', 1.5)};  // Risk/Reward ratio
input bool     UseTrailingStop = {str(rm.get('trailing_stop', False)).lower()};    // Use trailing stop
input bool     UseBreakEven = {str(rm.get('break_even', False)).lower()};       // Use break-even

"""
        
        # Add indicator parameters
        code += 'input group "=== Indicator Settings ==="\n'
        
        for ind in strategy.indicators:
            for param_name, param_value in ind.parameters.items():
                input_name = f"{ind.name}_{param_name}"
                if isinstance(param_value, float):
                    code += f"input double   {input_name} = {param_value};  // {ind.name} {param_name}\n"
                else:
                    code += f"input int      {input_name} = {param_value};  // {ind.name} {param_name}\n"
        
        code += "\n"
        return code
    
    def _generate_global_variables(self, strategy: GeneratedStrategy) -> str:
        """Generate global variables."""
        return """//--- Global Variables
CTrade trade;
int atrHandle;
double atrBuffer[];

datetime lastBarTime = 0;
bool isNewBar = false;

// Position tracking
bool hasOpenPosition = false;
int currentPositionType = -1;  // 0 = Buy, 1 = Sell

"""
    
    def _generate_indicator_handles(self, strategy: GeneratedStrategy) -> str:
        """Generate indicator handle declarations."""
        code = "//--- Indicator Handles\n"
        
        for i, ind in enumerate(strategy.indicators):
            code += f"int handle_{ind.name}_{i};\n"
            
            # Add buffers for each output column
            for col in ind.output_columns:
                col_clean = col.replace('{', '').replace('}', '').replace('period', '')
                code += f"double buffer_{ind.name}_{col_clean}_{i}[];\n"
        
        code += "\n"
        return code
    
    def _generate_oninit(self, strategy: GeneratedStrategy) -> str:
        """Generate OnInit function."""
        code = """//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize trade object
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   // Initialize ATR for risk management
   atrHandle = iATR(_Symbol, Timeframe, 14);
   if(atrHandle == INVALID_HANDLE)
   {
      Print("Failed to create ATR indicator");
      return(INIT_FAILED);
   }
   ArraySetAsSeries(atrBuffer, true);
   
"""
        
        # Initialize each indicator
        for i, ind in enumerate(strategy.indicators):
            code += f"   // Initialize {ind.name}\n"
            code += self._get_indicator_init_code(ind, i)
            
            # Set arrays as series
            for col in ind.output_columns:
                col_clean = col.replace('{', '').replace('}', '').replace('period', '')
                code += f"   ArraySetAsSeries(buffer_{ind.name}_{col_clean}_{i}, true);\n"
            
            code += "\n"
        
        code += """   Print("EA initialized successfully");
   return(INIT_SUCCEEDED);
}

"""
        return code
    
    def _get_indicator_init_code(self, ind: IndicatorConfig, index: int) -> str:
        """Generate initialization code for an indicator."""
        name = ind.name
        params = ind.parameters
        
        if name == "SMA":
            period = params.get('period', 20)
            return f"   handle_{name}_{index} = iMA(_Symbol, Timeframe, {name}_period, 0, MODE_SMA, PRICE_CLOSE);\n"
        
        elif name == "EMA":
            return f"   handle_{name}_{index} = iMA(_Symbol, Timeframe, {name}_period, 0, MODE_EMA, PRICE_CLOSE);\n"
        
        elif name == "MACD":
            return f"   handle_{name}_{index} = iMACD(_Symbol, Timeframe, {name}_fast_period, {name}_slow_period, {name}_signal_period, PRICE_CLOSE);\n"
        
        elif name == "RSI":
            return f"   handle_{name}_{index} = iRSI(_Symbol, Timeframe, {name}_period, PRICE_CLOSE);\n"
        
        elif name == "Stochastic":
            return f"   handle_{name}_{index} = iStochastic(_Symbol, Timeframe, {name}_k_period, {name}_d_period, 3, MODE_SMA, STO_LOWHIGH);\n"
        
        elif name == "BollingerBands":
            return f"   handle_{name}_{index} = iBands(_Symbol, Timeframe, {name}_period, 0, {name}_std_dev, PRICE_CLOSE);\n"
        
        elif name == "ATR":
            return f"   handle_{name}_{index} = iATR(_Symbol, Timeframe, {name}_period);\n"
        
        elif name == "ADX":
            return f"   handle_{name}_{index} = iADX(_Symbol, Timeframe, {name}_period);\n"
        
        elif name == "CCI":
            return f"   handle_{name}_{index} = iCCI(_Symbol, Timeframe, {name}_period, PRICE_TYPICAL);\n"
        
        elif name == "Williams_R":
            return f"   handle_{name}_{index} = iWPR(_Symbol, Timeframe, {name}_period);\n"
        
        elif name == "Momentum":
            return f"   handle_{name}_{index} = iMomentum(_Symbol, Timeframe, {name}_period, PRICE_CLOSE);\n"
        
        elif name == "OBV":
            return f"   handle_{name}_{index} = iOBV(_Symbol, Timeframe, VOLUME_TICK);\n"
        
        else:
            # Generic custom indicator
            return f"   handle_{name}_{index} = INVALID_HANDLE;  // TODO: Initialize {name}\n"
    
    def _generate_ondeinit(self) -> str:
        """Generate OnDeinit function."""
        return """//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Release indicator handles
   if(atrHandle != INVALID_HANDLE)
      IndicatorRelease(atrHandle);
   
   Print("EA deinitialized");
}

"""
    
    def _generate_ontick(self, strategy: GeneratedStrategy) -> str:
        """Generate OnTick function."""
        code = """//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new bar
   datetime currentBarTime = iTime(_Symbol, Timeframe, 0);
   isNewBar = (currentBarTime != lastBarTime);
   
   if(!isNewBar)
      return;
   
   lastBarTime = currentBarTime;
   
   // Update indicators
   if(!UpdateIndicators())
      return;
   
   // Check current position
   CheckCurrentPosition();
   
   // Manage existing position
   if(hasOpenPosition)
   {
      ManagePosition();
   }
   
   // Check for entry signals
   int signal = GetSignal();
   
   // Execute trades
   if(signal == 1 && !hasOpenPosition)  // Buy signal
   {
      OpenBuyPosition();
   }
   else if(signal == -1 && !hasOpenPosition)  // Sell signal
   {
      OpenSellPosition();
   }
}

"""
        return code
    
    def _generate_helper_functions(self, strategy: GeneratedStrategy) -> str:
        """Generate helper functions."""
        return """//+------------------------------------------------------------------+
//| Check current position                                             |
//+------------------------------------------------------------------+
void CheckCurrentPosition()
{
   hasOpenPosition = false;
   currentPositionType = -1;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            hasOpenPosition = true;
            currentPositionType = (int)PositionGetInteger(POSITION_TYPE);
            break;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate lot size                                                 |
//+------------------------------------------------------------------+
double CalculateLotSize(double stopLossPips)
{
   if(UseFixedLot)
      return LotSize;
   
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = accountBalance * (RiskPercent / 100.0);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double pointValue = tickValue / tickSize;
   
   double lots = riskAmount / (stopLossPips * pointValue);
   
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lots = MathFloor(lots / lotStep) * lotStep;
   lots = MathMax(minLot, MathMin(maxLot, lots));
   
   return lots;
}

//+------------------------------------------------------------------+
//| Get current ATR value                                              |
//+------------------------------------------------------------------+
double GetATR()
{
   if(CopyBuffer(atrHandle, 0, 0, 1, atrBuffer) <= 0)
      return 0;
   return atrBuffer[0];
}

"""
    
    def _generate_indicator_functions(self, strategy: GeneratedStrategy) -> str:
        """Generate indicator update functions."""
        code = """//+------------------------------------------------------------------+
//| Update all indicators                                              |
//+------------------------------------------------------------------+
bool UpdateIndicators()
{
   // Update ATR
   if(CopyBuffer(atrHandle, 0, 0, 3, atrBuffer) <= 0)
   {
      Print("Failed to copy ATR buffer");
      return false;
   }
   
"""
        
        for i, ind in enumerate(strategy.indicators):
            code += f"   // Update {ind.name}\n"
            code += self._get_indicator_copy_code(ind, i)
            code += "\n"
        
        code += """   return true;
}

"""
        return code
    
    def _get_indicator_copy_code(self, ind: IndicatorConfig, index: int) -> str:
        """Generate buffer copy code for an indicator."""
        name = ind.name
        code = ""
        
        # Number of buffers depends on indicator
        if name in ["MACD"]:
            code += f"   if(CopyBuffer(handle_{name}_{index}, 0, 0, 3, buffer_{name}_macd_{index}) <= 0) return false;\n"
            code += f"   if(CopyBuffer(handle_{name}_{index}, 1, 0, 3, buffer_{name}_macd_signal_{index}) <= 0) return false;\n"
            code += f"   if(CopyBuffer(handle_{name}_{index}, 2, 0, 3, buffer_{name}_macd_histogram_{index}) <= 0) return false;\n"
        elif name in ["Stochastic"]:
            code += f"   if(CopyBuffer(handle_{name}_{index}, 0, 0, 3, buffer_{name}_stoch_k_{index}) <= 0) return false;\n"
            code += f"   if(CopyBuffer(handle_{name}_{index}, 1, 0, 3, buffer_{name}_stoch_d_{index}) <= 0) return false;\n"
        elif name in ["BollingerBands"]:
            code += f"   if(CopyBuffer(handle_{name}_{index}, 0, 0, 3, buffer_{name}_bb_middle_{index}) <= 0) return false;\n"
            code += f"   if(CopyBuffer(handle_{name}_{index}, 1, 0, 3, buffer_{name}_bb_upper_{index}) <= 0) return false;\n"
            code += f"   if(CopyBuffer(handle_{name}_{index}, 2, 0, 3, buffer_{name}_bb_lower_{index}) <= 0) return false;\n"
        elif name in ["ADX"]:
            code += f"   if(CopyBuffer(handle_{name}_{index}, 0, 0, 3, buffer_{name}_adx_{index}) <= 0) return false;\n"
            code += f"   if(CopyBuffer(handle_{name}_{index}, 1, 0, 3, buffer_{name}_di_plus_{index}) <= 0) return false;\n"
            code += f"   if(CopyBuffer(handle_{name}_{index}, 2, 0, 3, buffer_{name}_di_minus_{index}) <= 0) return false;\n"
        else:
            # Single buffer indicators
            col = ind.output_columns[0] if ind.output_columns else name.lower()
            col_clean = col.replace('{', '').replace('}', '').replace('period', '')
            code += f"   if(CopyBuffer(handle_{name}_{index}, 0, 0, 3, buffer_{name}_{col_clean}_{index}) <= 0) return false;\n"
        
        return code
    
    def _generate_signal_functions(self, strategy: GeneratedStrategy) -> str:
        """Generate signal detection functions."""
        code = """//+------------------------------------------------------------------+
//| Get trading signal                                                 |
//+------------------------------------------------------------------+
int GetSignal()
{
   // Check entry conditions
   bool buySignal = CheckBuyConditions();
   bool sellSignal = CheckSellConditions();
   
   if(buySignal && !sellSignal)
      return 1;   // Buy
   if(sellSignal && !buySignal)
      return -1;  // Sell
   
   return 0;  // No signal
}

//+------------------------------------------------------------------+
//| Check buy conditions                                               |
//+------------------------------------------------------------------+
bool CheckBuyConditions()
{
"""
        
        # Generate buy conditions from entry rules
        buy_conditions = []
        for rule in strategy.entry_rules:
            if rule.signal_type == SignalType.BUY:
                for condition in rule.conditions:
                    mql_cond = self._condition_to_mql(condition, strategy)
                    if mql_cond:
                        buy_conditions.append(mql_cond)
        
        if buy_conditions:
            if len(buy_conditions) == 1:
                code += f"   return ({buy_conditions[0]});\n"
            else:
                code += f"   bool cond1 = ({buy_conditions[0]});\n"
                for i, cond in enumerate(buy_conditions[1:], 2):
                    code += f"   bool cond{i} = ({cond});\n"
                
                # Combine with AND
                conds = " && ".join([f"cond{i}" for i in range(1, len(buy_conditions) + 1)])
                code += f"   return ({conds});\n"
        else:
            code += "   return false;  // No buy conditions defined\n"
        
        code += """}

//+------------------------------------------------------------------+
//| Check sell conditions                                              |
//+------------------------------------------------------------------+
bool CheckSellConditions()
{
"""
        
        # Generate sell conditions
        sell_conditions = []
        for rule in strategy.entry_rules:
            if rule.signal_type == SignalType.SELL:
                for condition in rule.conditions:
                    mql_cond = self._condition_to_mql(condition, strategy)
                    if mql_cond:
                        sell_conditions.append(mql_cond)
        
        # Also check for opposite of buy conditions for sell
        if not sell_conditions:
            # Create opposite conditions
            for rule in strategy.entry_rules:
                if rule.signal_type == SignalType.BUY:
                    for condition in rule.conditions:
                        mql_cond = self._condition_to_mql(condition, strategy, invert=True)
                        if mql_cond:
                            sell_conditions.append(mql_cond)
        
        if sell_conditions:
            if len(sell_conditions) == 1:
                code += f"   return ({sell_conditions[0]});\n"
            else:
                code += f"   bool cond1 = ({sell_conditions[0]});\n"
                for i, cond in enumerate(sell_conditions[1:], 2):
                    code += f"   bool cond{i} = ({cond});\n"
                
                conds = " && ".join([f"cond{i}" for i in range(1, len(sell_conditions) + 1)])
                code += f"   return ({conds});\n"
        else:
            code += "   return false;  // No sell conditions defined\n"
        
        code += """}

"""
        return code
    
    def _condition_to_mql(self, condition: Condition, strategy: GeneratedStrategy, 
                          invert: bool = False) -> str:
        """Convert a condition to MQL5 code."""
        left = self._operand_to_mql(condition.left_operand, strategy)
        right = self._operand_to_mql(condition.right_operand, strategy)
        
        op = condition.operator
        
        # Invert operator if needed
        if invert:
            op_map = {
                ConditionOperator.GREATER_THAN: ConditionOperator.LESS_THAN,
                ConditionOperator.LESS_THAN: ConditionOperator.GREATER_THAN,
                ConditionOperator.GREATER_EQUAL: ConditionOperator.LESS_EQUAL,
                ConditionOperator.LESS_EQUAL: ConditionOperator.GREATER_EQUAL,
                ConditionOperator.CROSSES_ABOVE: ConditionOperator.CROSSES_BELOW,
                ConditionOperator.CROSSES_BELOW: ConditionOperator.CROSSES_ABOVE,
            }
            op = op_map.get(op, op)
        
        if op == ConditionOperator.GREATER_THAN:
            return f"{left} > {right}"
        elif op == ConditionOperator.LESS_THAN:
            return f"{left} < {right}"
        elif op == ConditionOperator.GREATER_EQUAL:
            return f"{left} >= {right}"
        elif op == ConditionOperator.LESS_EQUAL:
            return f"{left} <= {right}"
        elif op == ConditionOperator.EQUALS:
            return f"{left} == {right}"
        elif op == ConditionOperator.CROSSES_ABOVE:
            return f"({left} > {right} && {left.replace('[0]', '[1]')} <= {right.replace('[0]', '[1]')})"
        elif op == ConditionOperator.CROSSES_BELOW:
            return f"({left} < {right} && {left.replace('[0]', '[1]')} >= {right.replace('[0]', '[1]')})"
        
        return ""
    
    def _operand_to_mql(self, operand: str, strategy: GeneratedStrategy) -> str:
        """Convert an operand to MQL5 code."""
        # Check if it's a number
        try:
            float(operand)
            return operand
        except ValueError:
            pass
        
        # Check if it's a price column
        if operand == 'close':
            return "iClose(_Symbol, Timeframe, 0)"
        elif operand == 'open':
            return "iOpen(_Symbol, Timeframe, 0)"
        elif operand == 'high':
            return "iHigh(_Symbol, Timeframe, 0)"
        elif operand == 'low':
            return "iLow(_Symbol, Timeframe, 0)"
        
        # Check for indicator columns
        for i, ind in enumerate(strategy.indicators):
            for col in ind.output_columns:
                col_clean = col.replace('{', '').replace('}', '').replace('period', '')
                if operand == col or col_clean in operand:
                    return f"buffer_{ind.name}_{col_clean}_{i}[0]"
        
        # Default - return as variable name
        return operand.replace('.', '_').replace(' ', '_') + "[0]"
    
    def _generate_trading_functions(self, strategy: GeneratedStrategy) -> str:
        """Generate trading execution functions."""
        code = """//+------------------------------------------------------------------+
//| Open buy position                                                  |
//+------------------------------------------------------------------+
void OpenBuyPosition()
{
   double atr = GetATR();
   if(atr <= 0) return;
   
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double stopLoss = ask - (atr * ATRMultiplierSL);
   double takeProfit = ask + (atr * ATRMultiplierSL * RiskRewardRatio);
   
   double stopLossPips = (ask - stopLoss) / _Point;
   double lots = CalculateLotSize(stopLossPips);
   
   // Normalize prices
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   stopLoss = NormalizeDouble(stopLoss, digits);
   takeProfit = NormalizeDouble(takeProfit, digits);
   
   if(trade.Buy(lots, _Symbol, ask, stopLoss, takeProfit, "Auto Strategy Buy"))
   {
      Print("Buy order opened: Lots=", lots, " SL=", stopLoss, " TP=", takeProfit);
   }
   else
   {
      Print("Buy order failed: ", trade.ResultRetcode(), " - ", trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| Open sell position                                                 |
//+------------------------------------------------------------------+
void OpenSellPosition()
{
   double atr = GetATR();
   if(atr <= 0) return;
   
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double stopLoss = bid + (atr * ATRMultiplierSL);
   double takeProfit = bid - (atr * ATRMultiplierSL * RiskRewardRatio);
   
   double stopLossPips = (stopLoss - bid) / _Point;
   double lots = CalculateLotSize(stopLossPips);
   
   // Normalize prices
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   stopLoss = NormalizeDouble(stopLoss, digits);
   takeProfit = NormalizeDouble(takeProfit, digits);
   
   if(trade.Sell(lots, _Symbol, bid, stopLoss, takeProfit, "Auto Strategy Sell"))
   {
      Print("Sell order opened: Lots=", lots, " SL=", stopLoss, " TP=", takeProfit);
   }
   else
   {
      Print("Sell order failed: ", trade.ResultRetcode(), " - ", trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| Manage open position                                               |
//+------------------------------------------------------------------+
void ManagePosition()
{
   if(!hasOpenPosition) return;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetString(POSITION_SYMBOL) != _Symbol || 
            PositionGetInteger(POSITION_MAGIC) != MagicNumber)
            continue;
         
         double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
         double currentSL = PositionGetDouble(POSITION_SL);
         double currentTP = PositionGetDouble(POSITION_TP);
         double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
         
         double atr = GetATR();
         if(atr <= 0) continue;
         
         int posType = (int)PositionGetInteger(POSITION_TYPE);
         ulong ticket = PositionGetTicket(i);
         
         // Break-even management
         if(UseBreakEven)
         {
            double breakEvenTrigger = atr * ATRMultiplierSL * 0.5;
            
            if(posType == POSITION_TYPE_BUY)
            {
               if(currentPrice - openPrice >= breakEvenTrigger && currentSL < openPrice)
               {
                  double newSL = openPrice + _Point * 10;
                  trade.PositionModify(ticket, newSL, currentTP);
               }
            }
            else if(posType == POSITION_TYPE_SELL)
            {
               if(openPrice - currentPrice >= breakEvenTrigger && currentSL > openPrice)
               {
                  double newSL = openPrice - _Point * 10;
                  trade.PositionModify(ticket, newSL, currentTP);
               }
            }
         }
         
         // Trailing stop management
         if(UseTrailingStop)
         {
            double trailDistance = atr * ATRMultiplierSL;
            
            if(posType == POSITION_TYPE_BUY)
            {
               double newSL = currentPrice - trailDistance;
               if(newSL > currentSL && newSL < currentPrice)
               {
                  int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
                  newSL = NormalizeDouble(newSL, digits);
                  trade.PositionModify(ticket, newSL, currentTP);
               }
            }
            else if(posType == POSITION_TYPE_SELL)
            {
               double newSL = currentPrice + trailDistance;
               if(newSL < currentSL && newSL > currentPrice)
               {
                  int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
                  newSL = NormalizeDouble(newSL, digits);
                  trade.PositionModify(ticket, newSL, currentTP);
               }
            }
         }
         
         // Check exit conditions
         if(CheckExitConditions(posType))
         {
            trade.PositionClose(ticket);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check exit conditions                                              |
//+------------------------------------------------------------------+
bool CheckExitConditions(int positionType)
{
   // Get opposite signal as exit
   int signal = GetSignal();
   
   if(positionType == POSITION_TYPE_BUY && signal == -1)
      return true;
   if(positionType == POSITION_TYPE_SELL && signal == 1)
      return true;
   
   return false;
}

//+------------------------------------------------------------------+
"""
        return code
    
    def export_batch(self, strategies: List[GeneratedStrategy], 
                     output_dir: str) -> List[str]:
        """Export multiple strategies."""
        output_paths = []
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for strategy in strategies:
            filename = f"{strategy.name}.mq5"
            output_path = str(Path(output_dir) / filename)
            self.export(strategy, output_path)
            output_paths.append(output_path)
        
        return output_paths


def export_strategy_to_mql5(strategy: GeneratedStrategy, 
                            output_path: str = None) -> str:
    """Convenience function to export a strategy to MQL5."""
    exporter = MQL5Exporter()
    return exporter.export(strategy, output_path)
