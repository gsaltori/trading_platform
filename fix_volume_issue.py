#!/usr/bin/env python3
"""
Arreglar el problema de la columna 'volume' en ML engine
Los datos de MT5 usan 'tick_volume' en lugar de 'volume'
"""

def fix_volume_column():
    """Hacer que el feature engineering maneje ambos tipos de volumen"""
    
    file_path = 'ml/ml_engine.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar la sección de volumen en create_technical_features
    old_volume_code = """        # Volumen
        df['volume_sma'] = ta.SMA(df['volume'], timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_price_trend'] = df['volume'] * df['returns']"""
    
    new_volume_code = """        # Volumen (manejar tanto 'volume' como 'tick_volume')
        volume_col = 'volume' if 'volume' in df.columns else 'tick_volume'
        if volume_col in df.columns:
            df['volume_sma'] = ta.SMA(df[volume_col], timeperiod=20)
            df['volume_ratio'] = df[volume_col] / df['volume_sma']
            df['volume_price_trend'] = df[volume_col] * df['returns']
        else:
            # Si no hay datos de volumen, crear columnas dummy
            df['volume_sma'] = 1.0
            df['volume_ratio'] = 1.0
            df['volume_price_trend'] = 0.0"""
    
    if old_volume_code in content:
        content = content.replace(old_volume_code, new_volume_code)
        print("✅ Sección de volumen actualizada en create_technical_features")
    else:
        print("⚠️  No se encontró el patrón exacto, intentando alternativa...")
        # Patrón más flexible
        import re
        
        # Buscar y reemplazar la línea df['volume_sma']
        pattern1 = r"df\['volume_sma'\] = ta\.SMA\(df\['volume'\], timeperiod=20\)"
        replacement1 = "volume_col = 'volume' if 'volume' in df.columns else 'tick_volume'\n        df['volume_sma'] = ta.SMA(df[volume_col], timeperiod=20) if volume_col in df.columns else 1.0"
        content = re.sub(pattern1, replacement1, content)
        
        # Buscar y reemplazar la línea df['volume_ratio']
        pattern2 = r"df\['volume_ratio'\] = df\['volume'\] / df\['volume_sma'\]"
        replacement2 = "df['volume_ratio'] = df[volume_col] / df['volume_sma'] if volume_col in df.columns else 1.0"
        content = re.sub(pattern2, replacement2, content)
        
        # Buscar y reemplazar la línea df['volume_price_trend']
        pattern3 = r"df\['volume_price_trend'\] = df\['volume'\] \* df\['returns'\]"
        replacement3 = "df['volume_price_trend'] = df[volume_col] * df['returns'] if volume_col in df.columns else 0.0"
        content = re.sub(pattern3, replacement3, content)
        
        print("✅ Aplicados parches flexibles para volumen")
    
    # También arreglar en create_lagged_features donde se menciona 'volume'
    old_lagged = """        # Crear características retrasadas
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        df = self.create_lagged_features(df, price_columns, [1, 2, 3, 5, 10])"""
    
    new_lagged = """        # Crear características retrasadas
        price_columns = ['open', 'high', 'low', 'close']
        # Agregar columna de volumen si existe
        volume_col = 'volume' if 'volume' in df.columns else 'tick_volume'
        if volume_col in df.columns:
            price_columns.append(volume_col)
        df = self.create_lagged_features(df, price_columns, [1, 2, 3, 5, 10])"""
    
    if old_lagged in content:
        content = content.replace(old_lagged, new_lagged)
        print("✅ create_lagged_features actualizado")
    
    # También arreglar en create_rolling_features
    old_rolling = """        # Crear características rolling
        df = self.create_rolling_features(df, ['returns', 'volume', 'atr'], [5, 10, 20])"""
    
    new_rolling = """        # Crear características rolling
        rolling_columns = ['returns', 'atr']
        volume_col = 'volume' if 'volume' in df.columns else 'tick_volume'
        if volume_col in df.columns:
            rolling_columns.insert(1, volume_col)
        df = self.create_rolling_features(df, rolling_columns, [5, 10, 20])"""
    
    if old_rolling in content:
        content = content.replace(old_rolling, new_rolling)
        print("✅ create_rolling_features actualizado")
    
    # Escribir cambios
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Archivo {file_path} actualizado correctamente")
    print("\nCambios realizados:")
    print("  1. ✅ Manejo flexible de 'volume' vs 'tick_volume'")
    print("  2. ✅ Valores por defecto si no hay volumen")
    print("  3. ✅ Compatibilidad con datos de MT5")

if __name__ == "__main__":
    fix_volume_column()