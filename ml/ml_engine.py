# ml/ml_engine.py
"""
Machine Learning Engine for the trading platform.

Provides feature engineering, model training, and prediction capabilities
with support for multiple ML algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.cluster import KMeans
import joblib
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import optional dependencies
XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.info("XGBoost not available")

LIGHTGBM_AVAILABLE = False
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.info("LightGBM not available")

TENSORFLOW_AVAILABLE = False
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.info("TensorFlow not available")

# Try to import technical analysis library (prefer ta over talib)
TA_AVAILABLE = False
try:
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD, EMAIndicator, ADXIndicator, SMAIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    TA_AVAILABLE = True
except ImportError:
    logger.info("ta library not available, using basic indicators")


@dataclass
class MLModelConfig:
    """Configuration for ML models."""
    model_type: str  # 'classification', 'regression', 'clustering'
    algorithm: str   # 'random_forest', 'xgboost', 'lstm', 'ensemble'
    features: List[str]
    target: str
    lookback_window: int = 50
    prediction_horizon: int = 1
    train_test_split: float = 0.8
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLResult:
    """Results from ML model training."""
    model_name: str
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    actuals: np.ndarray
    metrics: Dict[str, float]
    model: Any
    feature_importance: Optional[pd.DataFrame] = None
    classification_report: Optional[Dict] = None


class FeatureEngineer:
    """Advanced feature engineering engine."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.fitted = False
        self.feature_columns = []
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced technical features."""
        df = data.copy()
        
        # Price and returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = df['high'] - df['low']
        df['normalized_range'] = df['price_range'] / df['close']
        
        # Moving averages and crosses
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            df[f'sma_ratio_{window}'] = df['close'] / df[f'sma_{window}'] - 1
        
        # Use ta library if available, otherwise basic calculations
        if TA_AVAILABLE:
            # RSI multiple timeframes
            for period in [7, 14, 21]:
                try:
                    df[f'rsi_{period}'] = RSIIndicator(df['close'], window=period).rsi()
                except Exception:
                    df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
            
            # MACD
            try:
                macd = MACD(df['close'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_histogram'] = macd.macd_diff()
            except Exception:
                df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
            
            # Bollinger Bands
            try:
                bb = BollingerBands(df['close'])
                df['bb_upper'] = bb.bollinger_hband()
                df['bb_lower'] = bb.bollinger_lband()
                df['bb_middle'] = bb.bollinger_mavg()
            except Exception:
                df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger(df['close'])
            
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Stochastic
            try:
                stoch = StochasticOscillator(df['high'], df['low'], df['close'])
                df['stoch_k'] = stoch.stoch()
                df['stoch_d'] = stoch.stoch_signal()
            except Exception:
                df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df['high'], df['low'], df['close'])
            
            # ADX
            try:
                adx = ADXIndicator(df['high'], df['low'], df['close'])
                df['adx'] = adx.adx()
                df['di_plus'] = adx.adx_pos()
                df['di_minus'] = adx.adx_neg()
            except Exception:
                df['adx'] = self._calculate_adx(df['high'], df['low'], df['close'])
                df['di_plus'] = 0
                df['di_minus'] = 0
            
            # ATR
            try:
                atr = AverageTrueRange(df['high'], df['low'], df['close'])
                df['atr'] = atr.average_true_range()
            except Exception:
                df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'])
        else:
            # Basic calculations without ta library
            df['rsi_14'] = self._calculate_rsi(df['close'], 14)
            df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger(df['close'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'])
        
        df['atr_pct'] = df['atr'] / df['close']
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_price_trend'] = df['volume'] * df['returns']
        
        # Volatility
        for window in [10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Price patterns
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & 
                          (df['low'] > df['low'].shift(1))).astype(int)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI without external library."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD without external library."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_bollinger(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0):
        """Calculate Bollinger Bands without external library."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                             k_period: int = 14, d_period: int = 3):
        """Calculate Stochastic Oscillator without external library."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        return k, d
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ATR without external library."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate simplified ADX."""
        # Simplified - just return volatility-based proxy
        tr = self._calculate_atr(high, low, close, period)
        return tr.rolling(window=period).mean() / close * 100
    
    def create_lagged_features(self, data: pd.DataFrame, columns: List[str], 
                              lags: List[int]) -> pd.DataFrame:
        """Create lagged features."""
        df = data.copy()
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df
    
    def create_rolling_features(self, data: pd.DataFrame, columns: List[str], 
                               windows: List[int]) -> pd.DataFrame:
        """Create rolling statistical features."""
        df = data.copy()
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_roll_mean_{window}'] = df[col].rolling(window).mean()
                    df[f'{col}_roll_std_{window}'] = df[col].rolling(window).std()
                    df[f'{col}_roll_min_{window}'] = df[col].rolling(window).min()
                    df[f'{col}_roll_max_{window}'] = df[col].rolling(window).max()
        return df
    
    def create_target_variable(self, data: pd.DataFrame, horizon: int = 1, 
                              method: str = 'classification') -> pd.Series:
        """Create target variable for ML."""
        if method == 'classification':
            # Classification: 1 if price goes up, 0 if down
            future_returns = data['close'].shift(-horizon) / data['close'] - 1
            target = (future_returns > 0).astype(int)
        elif method == 'regression':
            # Regression: future return
            target = data['close'].shift(-horizon) / data['close'] - 1
        else:
            raise ValueError("Method must be 'classification' or 'regression'")
        
        return target
    
    def prepare_features(self, data: pd.DataFrame, target: pd.Series, 
                        fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for ML with proper data handling."""
        # Create technical features
        df = self.create_technical_features(data)
        
        # Create lagged features (reduced for smaller datasets)
        price_columns = ['close', 'returns']
        available_cols = [col for col in price_columns if col in df.columns]
        df = self.create_lagged_features(df, available_cols, [1, 2, 3])
        
        # Create rolling features (reduced windows)
        roll_cols = ['returns']
        available_roll = [col for col in roll_cols if col in df.columns]
        df = self.create_rolling_features(df, available_roll, [5, 10])
        
        # Remove non-numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Align with target and remove NaN
        aligned_data = pd.concat([numeric_df, target.rename('target')], axis=1).dropna()
        
        if len(aligned_data) < 50:
            raise ValueError(f"Not enough data after preprocessing: {len(aligned_data)} rows")
        
        X = aligned_data.drop(columns=['target'])
        y = aligned_data['target']
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN with column means
        X = X.fillna(X.mean())
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), y


class MLEngine:
    """Main Machine Learning engine."""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.results = {}
    
    def train_model(self, data: pd.DataFrame, config: MLModelConfig) -> MLResult:
        """Train ML model."""
        logger.info(f"Training model {config.algorithm} for {config.target}")
        
        try:
            # Create target
            target = self.feature_engineer.create_target_variable(
                data, config.prediction_horizon, config.model_type
            )
            
            # Prepare features
            X, y = self.feature_engineer.prepare_features(data, target, fit=True)
            
            # Temporal split (no shuffle for time series)
            split_idx = int(len(X) * config.train_test_split)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            if len(X_train) < 30 or len(X_test) < 10:
                raise ValueError("Not enough data for training/testing split")
            
            # Create and train model
            model = self._create_model(config)
            
            if config.algorithm == 'lstm' and TENSORFLOW_AVAILABLE:
                # Prepare data for LSTM
                X_train_3d = self._reshape_for_lstm(X_train.values, config.lookback_window)
                X_test_3d = self._reshape_for_lstm(X_test.values, config.lookback_window)
                
                y_train_lstm = y_train.iloc[config.lookback_window:]
                y_test_lstm = y_test.iloc[config.lookback_window:]
                
                if len(X_train_3d) < 10:
                    raise ValueError("Not enough data for LSTM training")
                
                # Train LSTM
                model.fit(
                    X_train_3d, y_train_lstm,
                    validation_data=(X_test_3d, y_test_lstm),
                    epochs=config.parameters.get('epochs', 50),
                    batch_size=config.parameters.get('batch_size', 32),
                    verbose=0,
                    callbacks=[
                        EarlyStopping(patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(patience=5, factor=0.5)
                    ]
                )
                
                predictions_raw = model.predict(X_test_3d)
                predictions = (predictions_raw > 0.5).astype(int).flatten()
                probabilities = predictions_raw
                y_test = y_test_lstm
            else:
                # Traditional models
                model.fit(X_train, y_train)
                
                if hasattr(model, 'predict_proba') and config.model_type == 'classification':
                    probabilities = model.predict_proba(X_test)
                    predictions = model.predict(X_test)
                else:
                    predictions = model.predict(X_test)
                    probabilities = None
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, predictions, probabilities, config.model_type)
            
            # Feature importance
            feature_importance = self._get_feature_importance(model, X_train, config.algorithm)
            
            # Classification report
            classification_report_dict = None
            if config.model_type == 'classification':
                try:
                    classification_report_dict = classification_report(
                        y_test, predictions, output_dict=True, zero_division=0
                    )
                except Exception:
                    pass
            
            # Save results
            result = MLResult(
                model_name=f"{config.algorithm}_{config.target}",
                predictions=predictions,
                probabilities=probabilities,
                actuals=y_test.values,
                metrics=metrics,
                model=model,
                feature_importance=feature_importance,
                classification_report=classification_report_dict
            )
            
            self.models[result.model_name] = model
            self.results[result.model_name] = result
            
            logger.info(f"Model {result.model_name} trained. Accuracy: {metrics.get('accuracy', 0):.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def _create_model(self, config: MLModelConfig) -> Any:
        """Create model based on configuration."""
        params = config.parameters
        
        if config.algorithm == 'random_forest':
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                random_state=params.get('random_state', 42),
                n_jobs=-1
            )
        
        elif config.algorithm == 'xgboost':
            if not XGBOOST_AVAILABLE:
                logger.warning("XGBoost not available, using Random Forest")
                return RandomForestClassifier(n_estimators=100, random_state=42)
            return xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=params.get('random_state', 42),
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        elif config.algorithm == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                logger.warning("LightGBM not available, using Random Forest")
                return RandomForestClassifier(n_estimators=100, random_state=42)
            return lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', -1),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=params.get('random_state', 42),
                verbose=-1
            )
        
        elif config.algorithm == 'svm':
            return SVC(
                C=params.get('C', 1.0),
                kernel=params.get('kernel', 'rbf'),
                probability=True,
                random_state=params.get('random_state', 42)
            )
        
        elif config.algorithm == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=params.get('hidden_layer_sizes', (100, 50)),
                activation=params.get('activation', 'relu'),
                learning_rate_init=params.get('learning_rate', 0.001),
                random_state=params.get('random_state', 42),
                max_iter=params.get('max_iter', 1000)
            )
        
        elif config.algorithm == 'lstm':
            if not TENSORFLOW_AVAILABLE:
                logger.warning("TensorFlow not available, using MLP")
                return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            
            n_features = len(self.feature_engineer.feature_columns) if self.feature_engineer.feature_columns else 50
            
            model = Sequential([
                LSTM(units=params.get('lstm_units', 50), 
                     return_sequences=True, 
                     input_shape=(config.lookback_window, n_features)),
                Dropout(params.get('dropout_rate', 0.2)),
                LSTM(units=params.get('lstm_units', 50), return_sequences=False),
                Dropout(params.get('dropout_rate', 0.2)),
                Dense(units=params.get('dense_units', 25), activation='relu'),
                Dense(1, activation='sigmoid' if config.model_type == 'classification' else 'linear')
            ])
            
            optimizer = Adam(learning_rate=params.get('learning_rate', 0.001))
            loss = 'binary_crossentropy' if config.model_type == 'classification' else 'mse'
            
            model.compile(optimizer=optimizer, loss=loss, 
                         metrics=['accuracy'] if config.model_type == 'classification' else ['mae'])
            return model
        
        elif config.algorithm == 'ensemble':
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
            ]
            
            if XGBOOST_AVAILABLE:
                estimators.append(('xgb', xgb.XGBClassifier(
                    n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'
                )))
            
            return VotingClassifier(estimators=estimators, voting='soft')
        
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")
    
    def _reshape_for_lstm(self, X: np.ndarray, lookback: int) -> np.ndarray:
        """Reshape data for LSTM."""
        X_3d = []
        for i in range(lookback, len(X)):
            X_3d.append(X[i-lookback:i, :])
        return np.array(X_3d)
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          probabilities: Optional[np.ndarray], model_type: str) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred).flatten()
        
        if model_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            if probabilities is not None and len(probabilities.shape) > 1:
                try:
                    probs = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities.flatten()
                    metrics['log_loss'] = -np.mean(
                        y_true * np.log(probs + 1e-15) + 
                        (1 - y_true) * np.log(1 - probs + 1e-15)
                    )
                except Exception:
                    pass
        else:  # regression
            errors = y_true - y_pred
            metrics = {
                'mse': np.mean(errors ** 2),
                'rmse': np.sqrt(np.mean(errors ** 2)),
                'mae': np.mean(np.abs(errors)),
                'r2': 1 - (np.sum(errors ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            }
        
        return metrics
    
    def _get_feature_importance(self, model: Any, X: pd.DataFrame, 
                               algorithm: str) -> Optional[pd.DataFrame]:
        """Get feature importance."""
        try:
            if algorithm in ['random_forest', 'xgboost', 'lightgbm']:
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    return importance_df
            return None
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            return None
    
    def predict(self, model_name: str, data: pd.DataFrame) -> np.ndarray:
        """Make predictions with trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Create dummy target for feature preparation
        dummy_target = pd.Series(index=data.index, data=0)
        X_prepared, _ = self.feature_engineer.prepare_features(data, dummy_target, fit=False)
        
        if hasattr(model, 'predict'):
            return model.predict(X_prepared)
        else:
            raise ValueError("Model has no predict method")
    
    def save_model(self, model_name: str, filepath: str):
        """Save model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        joblib.dump({
            'model': self.models[model_name],
            'feature_engineer': self.feature_engineer,
            'result': self.results[model_name]
        }, filepath)
        
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """Load model from disk."""
        saved_data = joblib.load(filepath)
        
        self.models[model_name] = saved_data['model']
        self.feature_engineer = saved_data['feature_engineer']
        self.results[model_name] = saved_data['result']
        
        logger.info(f"Model {model_name} loaded from {filepath}")


class MarketRegimeDetector:
    """Market regime detector using ML clustering."""
    
    def __init__(self):
        self.ml_engine = MLEngine()
        self.regime_model = None
    
    def detect_regimes(self, data: pd.DataFrame, n_regimes: int = 3) -> pd.Series:
        """Detect market regimes using clustering."""
        # Create features for regime detection
        features = pd.DataFrame()
        
        # Volatility
        features['volatility'] = data['close'].pct_change().rolling(20).std()
        
        # Trend
        sma50 = data['close'].rolling(50).mean()
        features['trend'] = data['close'] / sma50 - 1
        
        # Range
        features['range'] = (data['high'] - data['low']) / data['close']
        
        # Volume (if available)
        if 'volume' in data.columns:
            vol_mean = data['volume'].rolling(20).mean()
            vol_std = data['volume'].rolling(20).std()
            features['volume_z'] = (data['volume'] - vol_mean) / vol_std
        
        features = features.dropna()
        
        if len(features) < n_regimes * 10:
            raise ValueError("Not enough data for regime detection")
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        regimes = kmeans.fit_predict(features_scaled)
        
        # Map clusters to interpretable regimes
        regime_map = self._interpret_regimes(features, regimes, n_regimes)
        
        return pd.Series(regime_map, index=features.index)
    
    def _interpret_regimes(self, features: pd.DataFrame, regimes: np.ndarray, n_regimes: int) -> List[str]:
        """Interpret clusters as market regimes."""
        regime_names = []
        
        for i in range(n_regimes):
            cluster_data = features[regimes == i]
            
            avg_volatility = cluster_data['volatility'].mean()
            avg_trend = cluster_data['trend'].mean()
            
            vol_threshold = features['volatility'].quantile(0.7)
            
            if avg_volatility > vol_threshold:
                if abs(avg_trend) > 0.02:
                    regime = "Volatile Trending"
                else:
                    regime = "Volatile Ranging"
            else:
                if abs(avg_trend) > 0.01:
                    regime = "Quiet Trending"
                else:
                    regime = "Quiet Ranging"
            
            regime_names.append(regime)
        
        return [regime_names[regime] for regime in regimes]
