import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QMenu, QPushButton, QWidget, QSizePolicy
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QPoint, QSize, QRect
import json, socket, sys, os.path, io, pickle, time, pandas
from canvas import GraphCanvas
from layout_colorwidget import Color
from error_box import ErrorBox
from main_fonts import (
    FontH,
    FontH_s,
    FontC
)

class DropMenu(QWidget):
    signal_set_feats = pyqtSignal(bool)

    def __init__(self, parent = None, settings = None):
        super().__init__(parent)

        # create the menu
        self.menu = QMenu(self)

        # the push button to show the menu
        self.option = QPushButton("Set Features",self)
        self.option.setMenu(self.menu)

        # action of setting default features
        self.default_settings = QAction("Default", self)

        # set the proper action corresponding
        self.menu.addAction(self.default_settings)

        self.default_settings.triggered.connect(self.put_default)
        
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        
        font_1 = FontH_s(FontH())

        self.setMinimumSize(310, 30)

        self.setFont(font_1)
        self.default_settings.setFont(font_1)

    pyqtSlot()
    def put_default(self):
        self.signal_set_feats.emit(True)

