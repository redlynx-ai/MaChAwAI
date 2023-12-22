import sys
from PyQt6.QtWidgets import (
    QMainWindow, QApplication,
    QLabel, QCheckBox, QComboBox, QListWidget, QLineEdit,
    QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QWidget, QMessageBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QSize, Qt, pyqtSignal, pyqtSlot

class ErrorBox(QMessageBox):
    def __init__(self, parent = None, errno = -1, msg = ""):
        super().__init__(parent)

        error = [
            'Missing file',
            'Missing Features',
            'No address',
            'Wrong address',
            'No port',
            'Wrong port',
            'No connection',
            'Wrong files',
            'No files',
            'No data',
            'Wrong data',
            'Close the connection',
            'Generic error'
        ]

        if (errno >= 0):
            self.about(self, "ERROR", error[errno])
        elif (msg != ""):
            self.about(self, "ERROR", msg)
