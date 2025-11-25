#!/usr/bin/env python3
"""
Agregar la clase MLEngine faltante a ml_engine.py
"""

import os
import shutil
from datetime import datetime

def add_mlengine_class():
    """Agregar la clase MLEngine al archivo"""
    
    filepath = "ml/ml_engine.py"
    
    if not os.path.exists(filepath):
        print(f"❌ Archivo no encontrado: {filepath}")
        return False
    
    print(f"🔧 Agregando clase MLEngine a {filepath}...")
    
    # Crear backup
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"   ✓ Backup: {backup_path}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar si MLEngine ya existe
        if 'class MLEngine:' in content or 'class MLEngine(' in content:
            print("   ⚠️  MLEngine ya existe en el archivo")
            print("   Verificando si está completa...")
            
            # Buscar dónde está
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if 'class MLEngine' in line:
                    print(f"   📍 Encontrada en línea {i}")
                    # Ver las siguientes 5 líneas
                    for j in range(i, min(i+5, len(lines))):
                        print(f"      {j}: {lines[j-1][:80]}")
            return False
        
        # Buscar dónde insertar (después de FeatureEngineer)
        insert_after = None
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if 'class FeatureEngineer:' in line or 'class FeatureEngineer(' in line:
                # Buscar el final de esta clase (siguiente 'class' o final de archivo)
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith('class ') and not lines[j].strip().startswith('#'):
                        insert_after = j
                        break
                if insert_after is None:
                    insert_after = len(lines)
                break
        
        if insert_after is None:
            print("   ❌ No se encontró dónde insertar MLEngine")
            return False
        
        print(f"   ✓ Insertando MLEngine después de línea {insert_after}")
        
        # Clase MLEngine completa
        mlengine_class = '''
class MLEngine:
    """Motor principal de Machine Learning"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.results = {}
        
    def train_model(self, data: pd.DataFrame, config: MLModelConfig) -> MLResult:
        """Entrenar modelo de ML"""
        logger.info(f"Entrenando modelo {config.algorithm} para {config.target}")
        
        try:
            # Crear target
            target = self.feature_engineer.create_target_variable(
                data, config.prediction_horizon, config.model_type
            )
            
            # Preparar características
            X, y = self.feature_engineer.prepare_features(data, target, fit=True)
            
            # Split temporal (no shuffle para series de tiempo)
            split_idx = int(len(X) * config.train_test_split)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Entrenar modelo
            model = self._create_model(config)
            
            if config.algorithm == 'lstm':
                # Preparar datos para LSTM
                X_train_3d = self._reshape_for_lstm(X_train.values, config.lookback_window)
                X_test_3d = self._reshape_for_lstm(X_test.values, config.lookback_window)
                
                # Ajustar y_train para LSTM
                y_train_lstm = y_train.iloc[config.lookback_window:]
                y_test_lstm = y_test.iloc[config.lookback_window:]
                
                # Entrenar LSTM
                history = model.fit(
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
                
                # Predecir
                predictions = model.predict(X_test_3d)
                probabilities = predictions if config.model_type == 'classification' else None
                
            else:
                # Modelos tradicionales
                model.fit(X_train, y_train)
                
                # Predecir
                if hasattr(model, 'predict_proba') and config.model_type == 'classification':
                    probabilities = model.predict_proba(X_test)
                    predictions = model.predict(X_test)
                else:
                    predictions = model.predict(X_test)
                    probabilities = None
            
            # Calcular métricas
            metrics = self._calculate_metrics(y_test, predictions, probabilities, config.model_type)
            
            # Importancia de características
            feature_importance = self._get_feature_importance(model, X_train, config.algorithm)
            
            # Reporte de clasificación
            classification_report_dict = None
            if config.model_type == 'classification':
                from sklearn.metrics import classification_report
                classification_report_dict = classification_report(
                    y_test, predictions, output_dict=True
                )
            
            # Guardar resultados
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
            
            logger.info(f"Modelo {result.model_name} entrenado. Accuracy: {metrics.get('accuracy', 0):.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error entrenando modelo: {e}")
            raise
    
    def _create_model(self, config: MLModelConfig):
        """Crear modelo según configuración"""
        params = config.parameters
        
        if config.algorithm == 'random_forest':
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                random_state=params.get('random_state', 42)
            )
            
        elif config.algorithm == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=params.get('random_state', 42)
            )
            
        elif config.algorithm == 'lightgbm':
            return lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', -1),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=params.get('random_state', 42)
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
            model = Sequential([
                LSTM(units=params.get('lstm_units', 50), 
                     return_sequences=True, 
                     input_shape=(config.lookback_window, len(config.features) if config.features else 10)),
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
            estimators = []
            estimators.append(('rf', RandomForestClassifier(n_estimators=100, random_state=42)))
            estimators.append(('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42)))
            return VotingClassifier(estimators=estimators, voting='soft')
        
        else:
            raise ValueError(f"Algoritmo no soportado: {config.algorithm}")
    
    def _reshape_for_lstm(self, X: np.ndarray, lookback: int) -> np.ndarray:
        """Reformatear datos para LSTM"""
        X_3d = []
        for i in range(lookback, len(X)):
            X_3d.append(X[i-lookback:i, :])
        return np.array(X_3d)
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          probabilities: Optional[np.ndarray], model_type: str) -> Dict[str, float]:
        """Calcular métricas de evaluación"""
        if model_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': precision_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            if probabilities is not None and len(probabilities.shape) > 1:
                metrics['log_loss'] = -np.mean(
                    y_true * np.log(probabilities[:, 1] + 1e-15) + 
                    (1 - y_true) * np.log(1 - probabilities[:, 1] + 1e-15)
                )
                
        else:  # regression
            errors = y_true - y_pred
            metrics = {
                'mse': np.mean(errors ** 2),
                'rmse': np.sqrt(np.mean(errors ** 2)),
                'mae': np.mean(np.abs(errors)),
                'r2': 1 - (np.sum(errors ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            }
        
        return metrics
    
    def _get_feature_importance(self, model, X: pd.DataFrame, 
                              algorithm: str) -> Optional[pd.DataFrame]:
        """Obtener importancia de características"""
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
            logger.warning(f"No se pudo obtener importancia de características: {e}")
            return None
    
    def predict(self, model_name: str, data: pd.DataFrame) -> np.ndarray:
        """Realizar predicciones con modelo entrenado"""
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        model = self.models[model_name]
        dummy_target = pd.Series(index=data.index, data=0)
        X_prepared, _ = self.feature_engineer.prepare_features(data, dummy_target, fit=False)
        
        if hasattr(model, 'predict'):
            return model.predict(X_prepared)
        else:
            raise ValueError("Modelo no tiene método predict")
    
    def save_model(self, model_name: str, filepath: str):
        """Guardar modelo en disco"""
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        joblib.dump({
            'model': self.models[model_name],
            'feature_engineer': self.feature_engineer,
            'result': self.results[model_name]
        }, filepath)
        
        logger.info(f"Modelo {model_name} guardado en {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """Cargar modelo desde disco"""
        saved_data = joblib.load(filepath)
        
        self.models[model_name] = saved_data['model']
        self.feature_engineer = saved_data['feature_engineer']
        self.results[model_name] = saved_data['result']
        
        logger.info(f"Modelo {model_name} cargado desde {filepath}")

'''
        
        # Insertar la clase
        lines.insert(insert_after, mlengine_class)
        
        # Guardar archivo
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Clase MLEngine agregada exitosamente")
        
        # Verificar
        with open(filepath, 'r', encoding='utf-8') as f:
            new_content = f.read()
        
        if 'class MLEngine:' in new_content:
            print(f"✅ Verificación: MLEngine presente en el archivo")
            return True
        else:
            print(f"❌ Verificación falló")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_import():
    """Verificar que ahora se puede importar"""
    print("\n🧪 Verificando import...")
    
    import sys
    # Limpiar cache
    if 'ml.ml_engine' in sys.modules:
        del sys.modules['ml.ml_engine']
    if 'ml' in sys.modules:
        del sys.modules['ml']
    
    try:
        from ml.ml_engine import MLEngine, MLModelConfig
        print("✅ Import exitoso!")
        print(f"   MLEngine: {MLEngine}")
        print(f"   MLModelConfig: {MLModelConfig}")
        return True
    except Exception as e:
        print(f"❌ Import falló: {e}")
        return False

def main():
    print("=" * 60)
    print("🔧 AGREGANDO CLASE MLEngine")
    print("=" * 60)
    print()
    
    if add_mlengine_class():
        if verify_import():
            print()
            print("=" * 60)
            print("✅ ¡ÉXITO TOTAL!")
            print("=" * 60)
            print()
            print("Ahora ejecuta:")
            print("  python -m pytest tests/test_suite.py -v")
            return 0
        else:
            print()
            print("⚠️  Clase agregada pero hay problemas de import")
            print("Intenta limpiar el cache:")
            print("  del /s /q ml\\__pycache__")
            print("  del /s /q __pycache__")
            return 1
    else:
        print()
        print("❌ No se pudo agregar la clase")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())