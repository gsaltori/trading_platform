#!/usr/bin/env python3
"""
Script de reparación rápida para ml_engine.py
"""

import os
import shutil
from datetime import datetime

def repair_ml_engine():
    """Reparar archivo ml_engine.py con el método prepare_features corregido"""
    
    filepath = "ml/ml_engine.py"
    
    if not os.path.exists(filepath):
        print(f"❌ Archivo no encontrado: {filepath}")
        return False
    
    print(f"🔧 Reparando {filepath}...")
    
    # Crear backup
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"   ✓ Backup creado: {backup_path}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Buscar el método prepare_features y reemplazarlo
        in_prepare_features = False
        indent_level = None
        new_lines = []
        skip_until_next_method = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Detectar inicio del método prepare_features
            if 'def prepare_features(self, data: pd.DataFrame, target: pd.Series' in line:
                print(f"   ✓ Encontrado prepare_features en línea {i+1}")
                in_prepare_features = True
                indent_level = len(line) - len(line.lstrip())
                skip_until_next_method = True
                
                # Insertar nuevo método
                new_method = '''    def prepare_features(self, data: pd.DataFrame, target: pd.Series, 
                        fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Preparar características para ML"""
        
        # Validar datos de entrada
        if len(data) < 100:
            raise ValueError(f"Insufficient data: {len(data)} rows. Need at least 100 rows.")
        
        # Crear características técnicas
        df = self.create_technical_features(data)
        
        # Crear características retrasadas (REDUCIDO para evitar demasiados NaN)
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        existing_price_columns = [col for col in price_columns if col in df.columns]
        df = self.create_lagged_features(df, existing_price_columns, [1, 2, 3])  # Solo 3 lags
        
        # Crear características rolling (REDUCIDO)
        rolling_columns = []
        if 'returns' in df.columns:
            rolling_columns.append('returns')
        if 'volume' in df.columns:
            rolling_columns.append('volume')
        if 'atr' in df.columns:
            rolling_columns.append('atr')
        
        if rolling_columns:
            df = self.create_rolling_features(df, rolling_columns, [5, 10])  # Solo 2 ventanas
        
        # Alinear con target y eliminar NaN
        aligned_data = pd.concat([df, target], axis=1).dropna()
        
        # Verificar que tenemos suficientes datos
        if len(aligned_data) < 10:
            raise ValueError(f"After removing NaN, only {len(aligned_data)} samples remain. "
                            f"Original data had {len(data)} rows. Need at least 10 samples.")
        
        X = aligned_data.iloc[:, :-1]
        y = aligned_data.iloc[:, -1]
        
        # Limpiar datos
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna(axis=1, how='all')
        X = X.loc[:, X.std() > 0]
        
        if X.shape[1] == 0:
            raise ValueError("No valid features after preprocessing")
        
        # Escalar características
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), y

'''
                new_lines.append(new_method)
                i += 1
                continue
            
            # Si estamos saltando el método viejo, buscar el siguiente método
            if skip_until_next_method:
                if line.strip().startswith('def ') and not 'def prepare_features' in line:
                    # Encontramos el siguiente método
                    skip_until_next_method = False
                    new_lines.append(line)
                # Si encontramos una nueva clase, también terminamos
                elif line.strip().startswith('class '):
                    skip_until_next_method = False
                    new_lines.append(line)
            else:
                new_lines.append(line)
            
            i += 1
        
        # Guardar archivo reparado
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print(f"✅ {filepath} reparado exitosamente")
        
        # Verificar sintaxis
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                compile(f.read(), filepath, 'exec')
            print(f"✅ Sintaxis verificada correctamente")
            return True
        except SyntaxError as e:
            print(f"❌ Error de sintaxis después de reparar: {e}")
            print(f"   Restaurando backup...")
            shutil.copy2(backup_path, filepath)
            return False
            
    except Exception as e:
        print(f"❌ Error durante reparación: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🔧 REPARACIÓN RÁPIDA DE ml_engine.py")
    print("=" * 60)
    print()
    
    if repair_ml_engine():
        print()
        print("=" * 60)
        print("✅ REPARACIÓN EXITOSA")
        print("=" * 60)
        print()
        print("📊 Ahora puedes ejecutar los tests:")
        print("   python -m pytest tests/test_suite.py -v")
        return 0
    else:
        print()
        print("=" * 60)
        print("❌ LA REPARACIÓN FALLÓ")
        print("=" * 60)
        print()
        print("💡 Soluciones alternativas:")
        print("   1. Restaura un backup manualmente")
        print("   2. Revisa ml/ml_engine.py línea por línea")
        print("   3. Compara con la versión original del documento")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())