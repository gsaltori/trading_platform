#!/usr/bin/env python3
"""
Arreglar detect_regimes para manejar tick_volume en lugar de volume
"""

def fix_detect_regimes():
    """Arreglar la función detect_regimes en MarketRegimeDetector"""
    
    file_path = 'ml/ml_engine.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar y reemplazar en detect_regimes
    old_volume_z = "        features['volume_z'] = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()"
    
    new_volume_z = """        # Volumen (manejar tick_volume si volume no existe)
        volume_col = 'volume' if 'volume' in data.columns else 'tick_volume'
        if volume_col in data.columns:
            features['volume_z'] = (data[volume_col] - data[volume_col].rolling(20).mean()) / data[volume_col].rolling(20).std()
        else:
            # Sin datos de volumen, usar 0
            features['volume_z'] = 0.0"""
    
    if old_volume_z in content:
        content = content.replace(old_volume_z, new_volume_z)
        print("✅ detect_regimes actualizado para manejar tick_volume")
    else:
        print("⚠️  Patrón no encontrado, buscando alternativa...")
        import re
        
        # Patrón más flexible
        pattern = r"features\['volume_z'\] = \(data\['volume'\] - data\['volume'\]\.rolling\(20\)\.mean\(\)\) / data\['volume'\]\.rolling\(20\)\.std\(\)"
        replacement = """volume_col = 'volume' if 'volume' in data.columns else 'tick_volume'
        if volume_col in data.columns:
            features['volume_z'] = (data[volume_col] - data[volume_col].rolling(20).mean()) / data[volume_col].rolling(20).std()
        else:
            features['volume_z'] = 0.0"""
        
        content = re.sub(pattern, replacement, content)
        print("✅ Aplicado parche flexible en detect_regimes")
    
    # Escribir cambios
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Archivo {file_path} actualizado correctamente")
    print("\nCambios realizados:")
    print("  1. ✅ MarketRegimeDetector ahora maneja 'tick_volume'")
    print("  2. ✅ Valores por defecto si no hay volumen")

if __name__ == "__main__":
    fix_detect_regimes()