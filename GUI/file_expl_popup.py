import sys
from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.QtGui import QDesktopServices, QMouseEvent
from PyQt6.QtCore import QUrl, Qt, QEvent
from main_fonts import FontH

HOME_PATH="/home/" # to change to C: if implementing in Windows

class ClickableLabel(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        font_1 = FontH()

        self.setFont(font_1)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_file_explorer()

    def enterEvent(self, event: QEvent) -> None:
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def leaveEvent(self, event: QEvent) -> None:
        self.unsetCursor()

    def open_file_explorer(self):
        file_dialog = QDesktopServices()
        file_dialog.openUrl(QUrl.fromLocalFile(HOME_PATH))


if __name__ == "__main__":
    app = QApplication(sys.argv)

    label = ClickableLabel("Click me to open file explorer")
    label.show()

    sys.exit(app.exec())

