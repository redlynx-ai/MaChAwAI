from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPalette, QColor, QPixmap
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QPalette, QColor, QPixmap
from PyQt6.QtCore import QSize

class Color(QWidget):
    def __init__(self,color : tuple):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)
        # self.setGeometry(50, 50, 50, 50)
        palette = self.palette()
        r, g, b = color
        palette.setColor(QPalette.ColorRole.Window, QColor(r, g, b))
        self.setPalette(palette)

