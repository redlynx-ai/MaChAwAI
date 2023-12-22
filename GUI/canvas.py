import sys
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap
from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QStackedLayout,
    QWidget,
    QPushButton
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from error_box import ErrorBox
from file_expl_popup import ClickableLabel

# TODO: file explorer popup clickable label
# TODO: (OPTIONAL) cleaner icon displays

class DropFiles(QWidget):
    signal_file_paths = pyqtSignal(str)
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.file_path = None

        # self.layout = QGridLayout()
        self.layout_files = QVBoxLayout()
        self.layout= QStackedLayout()

        self.label_to_conn = QLabel("Please connect to upload a file")
        self.label_to_conn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.label_file_exp = ClickableLabel("Click me to open file explorer")
        self.dropped_files = QWidget()
        self.layout.addWidget(self.label_to_conn)
        self.layout.addWidget(self.label_file_exp)
        self.layout.addWidget(self.dropped_files)

        self.setStyleSheet("background: #b4b4b4")

        self.setLayout(self.layout)
        
        # self.layout.setCurrentWidget(self.label_to_conn)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.layout.setCurrentWidget(self.dropped_files)
        if self.file_path:
            self.remove_icons()
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                self.file_path = url.toLocalFile()
                # self.display_file_icon(file_path)
                # print(file_path.split('/')[-1])
                if self.file_path.split('/')[-1].split('.')[-1] == "dat":
                    self.display_icon_path(self.file_path, 0)
                    self.echo_paths(self.file_path)
                else:
                    continue
            event.acceptProposedAction()

    def reset_layout(self):
        self.layout.setCurrentWidget(self.label_to_conn)
        self.setAcceptDrops(False)

    def display_icon_path(self, file_path, cosj = -1):

        print(file_path)

        icon_path = QWidget()
        ip_layout = QHBoxLayout()

        icon = "icons/file_icon.png"

        pixmap = QLabel("")
        pixmap.setPixmap( QPixmap(f"{icon}").scaledToWidth(30) )
        # pixmap.setPixmap( QPixmap(f"{icon}") )

        # label = QLabel(f".../{'/'.join(file_path.split('/')[-2:])}")
        label = QLabel(f"{file_path}")
        label.setFont(label.font())
        label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBaseline)

        ip_layout.addWidget(pixmap)
        ip_layout.addWidget(label)

        icon_path.setLayout(ip_layout)
        
        # row = self.layout().rowCount()
        self.layout_files.addWidget(icon_path)
        self.dropped_files.setLayout(self.layout_files)

    def process_dropped_files(self, file_path):
        # Handle the dropped file paths
        print("Dropped file:", file_path)
        print(type(file_path))

    def remove_icons(self):
        for i in range(self.layout_files.count()):
            widg = self.layout_files.itemAt(i).widget()
            widg.setParent(None)

    def set_accept(self, cond):
        if cond:
            self.layout.setCurrentWidget(self.dropped_files)
        else:
            self.layout.setCurrentWidget(self.label_file_exp)

    @pyqtSlot()
    def echo_paths(self, file_paths):
        self.signal_file_paths.emit(self.file_path)

class GraphCanvas(FigureCanvas):
    # there is no qtpy signal to initiate graph drawing
    # that is handled by an external widget
    def __init__(self, parent = None):
        self.figure = Figure()
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Displacement")
        self.ax.set_ylabel("Stress")
        self.ax.set_title("Simulated against Predicted values")

    def update_graph(self, x_values, y_values):
        """Update the graph with new data"""
        self.ax.clear()
        # self.ax.plot(x_values[0], y_values, label="simulation")
        self.ax.plot(x_values[1], y_values, label="real")
        self.ax.plot(x_values[2], y_values, label="predicted")
        self.ax.legend(loc="best")
        self.draw()

    def draw_graph(self, X, Y):
        self.update_graph(X, Y)

    def clear_graphs(self):
        self.figure.clear()
        self.ax.clear()
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)

if __name__ == "__main__":

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            self.setWindowTitle("File Selection")
            self.canvas = DropFiles()
            self.setCentralWidget(self.canvas)

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
