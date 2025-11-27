import sys
print(f"Python path: {sys.executable}")
print(f"Paths: {sys.path}")

try:
    from PyQt6.QtWidgets import QApplication, QLabel
    from PyQt6.QtCore import Qt
    
    app = QApplication([])
    label = QLabel("¡PyQt6 funciona!")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.show()
    app.exec()
    print("¡Todo funciona correctamente!")
    
except ImportError as e:
    print(f"Error de importación: {e}")