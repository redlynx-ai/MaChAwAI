from PyQt6.QtGui import QColor, QFont

class FontH(QFont):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFamily("Helvetica")
        self.setWeight(QFont.Weight.DemiBold)
        self.setPointSize(15)

class FontH_s(QFont):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPointSize(9)

class FontC(QFont):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFamily("Courier")
        self.setWeight(QFont.Weight.Light)
        self.setPointSize(8)
