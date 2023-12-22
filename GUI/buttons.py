import sys
from PyQt6.QtWidgets import (
    QSizePolicy,
    QStackedLayout,
    QWidget,
    QPushButton
)
from PyQt6.QtCore import pyqtSignal, pyqtSlot
import socket, sys, os.path, pickle, pandas
import json
from layout_colorwidget import Color
from error_box import ErrorBox
from main_fonts import (
    FontH
)
from pull_out_data import (
    extract_data,
    replace
)

HEADERSIZE = 10

# Connect Button
class ConnectButton(QPushButton):
    signal_set_socket = pyqtSignal(socket.socket)
    signal_set_port_n = pyqtSignal(int)

    def __init__(self, parent = None):

        super().__init__("Connect",parent)

        self.setSizePolicy( QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed )

        self.clicked.connect(self.connect_to_container)

        font_1 = FontH()

        self.setFont(font_1)
        
        self.setMinimumSize(310, 30)

    @pyqtSlot()
    def connect_to_container(self):
        
        # self.host = self.parent().parent().address
        # self.port = self.parent().parent().port
        self.host = "0.0.0.0"
        # self.host = 'togn3k.xyz'
        # self.host = '51.254.98.7'
        # self.port = 3000
        self.port = 8888

        # may Implement automatic solution

        if self.host and self.port:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.info()

            try:
                client_socket.connect((self.host, self.port))
                print(f"Connected to {self.host}:{self.port}")
            except Exception as e:
                print(e)

            self.signal_set_socket.emit(client_socket)
            self.signal_set_port_n.emit(self.port)
        else:
            ErrorBox(self, 1)

    def info(self):
        print(f"Address = {self.host}\nPort = {self.port}")

# Upload Button 
class UploadButton(QPushButton):

    signal_start_conn = pyqtSignal(bool)
    # unused
    signal_data_X = pyqtSignal(pandas.core.series.Series)
    signal_data_Y = pyqtSignal(pandas.core.series.Series)

    def __init__(self, parent = None):
        super().__init__("Upload", parent)

        self.clicked.connect(self.upload)

        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.path_to_csv = ''
        # unused
        # self.path_to_json = ''
        
        font_1 = FontH()

        self.setMinimumSize(310, 30)

        self.setFont(font_1)

    def package_data(self, path, feat):

        # path_to_csv = self.parent().parent().parent().parent().path_to_files[0]
        # self.path_to_json = self.parent().parent().path_to_files[1]
        # features = self.parent().parent().parent().parent().features
        path_to_file = path
        features = feat

        print(f"{path_to_file}\n{features}")

        data = None
        if (os.path.isfile(path_to_file)):
            try:
                # extract_data returns a np.ndarray with the 
                # properly truncated values
                files = {
                    'data' : extract_data(f"{path_to_file}"),
                    'json' : features
                }
                features = json.loads(features)
                files['data']['EXTS'] = files['data']['EXTS'] / features['LG']
                features['BP'] = len(files["data"]["LOAD"]) - 1
                # features = f'{features[:-1]}, "BP": {len(files["data"]["LOAD"]) - 1}' + '}'
                files['json'] = json.dumps(features)
                # files = json.dumps(files)
                data = pickle.dumps(files)
            except Exception as e:
                print(e)
        else:
            ErrorBox(self, 1)
        data = self.embed_header(data)

        return data

    def embed_header(self, data):
        if data:
            try:
                return bytes(f"{len(data):<{HEADERSIZE}}", "utf-8")+data
            except BytesWarning as warn:
                raise warn
        else:
            return bytes(f"{-1:<{HEADERSIZE}}", "utf-8")

    def send_data(self, data):
        self.socket = self.parent().parent().parent().parent().client_socket
        print(f"{self.socket}")

        try:
            self.socket.sendall(data)
        except OSError as e:
            print(e)

    def read_results(self):
        print('Reading results...')
        sample, pred = b'', b''
        new_msg = True
        msglen = sys.maxsize
        try:
            while len(sample)-HEADERSIZE < msglen:
                receive_len = min(msglen - len(sample) + HEADERSIZE, 1024)
                res = self.socket.recv(receive_len)

                if new_msg:
                    msglen=int(res[:HEADERSIZE])
                    new_msg = False
                sample += res

                receive_len = min(msglen - len(sample) + HEADERSIZE, HEADERSIZE)

            msglen = sys.maxsize
            new_msg = True
            
            while len(pred)-HEADERSIZE < msglen:
                receive_len = min(msglen - len(pred) + HEADERSIZE, 1024)
                res = self.socket.recv(receive_len)

                if new_msg:
                    msglen=int(res[:HEADERSIZE])
                    new_msg = False 
                pred += res

                receive_len = min(msglen - len(pred) + HEADERSIZE, HEADERSIZE)

        except OSError as e:
            raise e

        # self.signal_data_X.emit(pickle.loads(sample[HEADERSIZE:]))
        # self.signal_data_Y.emit(pickle.loads(pred[HEADERSIZE:]))
        
        return pickle.loads(sample[HEADERSIZE:]), pickle.loads(pred[HEADERSIZE:])

    @pyqtSlot()
    def upload(self):
        # Atomic Send: all of these funcs at once
        # data = self.package_data()
        # self.send_data(data)
        #
        # results = self.read_results()
        #
        # self.signal_data_X.emit(results[0])
        # self.signal_data_Y.emit(resutls[1])
        # May better to be all executed once certain socket and files are correctly uploaded (volatile container)
        self.signal_start_conn.emit(True)

class CloseButton(QPushButton):
    signal_end_conn = pyqtSignal(bool)

    def __init__(self, parent = None):
        super().__init__("Disconnect",parent)

        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.clicked.connect(self.drop_conn)
        font_1 = FontH()

        self.setFont(font_1)

        self.setMinimumSize(310, 30)

    def drop_conn(self):
        self.signal_end_conn.emit(True)

class StatusBar(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)

        statusLayout = QStackedLayout()

        self.setFixedSize(310, 5)

        self.mode = 0

        # statusLayout.setGeometry(QRect(QPoint(), QSize()))

        statusLayout.addWidget(Color((224, 17, 17)))
        statusLayout.addWidget(Color((224, 224, 17)))
        statusLayout.addWidget(Color((121, 224, 17)))

        self.setLayout(statusLayout)

    def set_status(self, mode):
        self.mode = mode
        if self.mode == 0: # not connected
            self.layout().setCurrentIndex(0)
        elif self.mode == 1: # connected
            self.layout().setCurrentIndex(1)
        elif self.mode > 1: # files succesfully uploaded and retrieved
            self.layout().setCurrentIndex(2)

    def get_status(self):
        return self.mode

# Show graph Button
class ShowGraphs(QPushButton):
    signal_get_data = pyqtSignal(bool)

    def __init__(self, parent = None, canvas_item = None):
        super().__init__("Show graph",parent)

        self.setSizePolicy( QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed )

        self.data_X = None
        self.data_Y = None

        self.canvas = canvas_item
        self.clicked.connect(self.request_data)
        font_1 = FontH()

        self.setFont(font_1)

        self.setMinimumSize(310, 30)

    def copy_data(self, x, y):
        self.data_X = x
        self.data_Y = y

    def set_canvas_item(self, canvas_item):
        self.canvas = canvas_item

    def update_input(self):
        if self.canvas:
            print(self.data_X, self.data_Y)
            self.canvas.draw_graph(self.data_X, self.data_Y)
        else:
            # raise ObjNotFoundError
            print("No canvas object!")

    @pyqtSlot(bool)
    def request_data(self):
        self.signal_get_data.emit(True)

# Update dat file
class UpdateDat(QPushButton):
    signal_update_dat = pyqtSignal(bool)

    def __init__(self, parent = None):
        super().__init__("Write to file", parent)

        self.setSizePolicy( QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed )

        font_1 = FontH()

        self.clicked.connect(self.request_update_dat)
        self.setFont(font_1)

        self.setMinimumSize(310, 30)

    def write_data_to_file(self, file_path, features, pred_values):
        replace(file_path, features, pred_values)

    @pyqtSlot(bool)
    def request_update_dat(self):
        self.signal_update_dat.emit(True)
