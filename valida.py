import sys
print(sys.path)
try:
    import PyQt6
    print("PyQt6 importado correctamente")
except ImportError as e:
    print(f"Error al importar: {e}")