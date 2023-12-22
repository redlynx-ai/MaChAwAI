# from mid_handlers import buttons, tabs, inputs
import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTabWidget
)
from canvas import DropFiles, GraphCanvas
from buttons import (
    ConnectButton,
    UploadButton,
    StatusBar,
    ShowGraphs,
    CloseButton,
    UpdateDat
)
# from input_boxes import AddressBox, PortBox
from error_box import ErrorBox
from form_features import FeaturesForm
from set_feat_menu import DropMenu
import os.path, json, socket
from PyQt6.QtCore import pyqtSlot, QSize

# TODO: implement macrostates translations
# TODO: delete unused function
# TODO: line things up, put some color, (OPTIONAL) fixed glitched colors when moving window
# TODO: (OPTIONAL) refactor and clean up code

class MainWindow(QMainWindow):

    # global vars
    
    macrostates = {
        'RED' : 0,
        'YLW' : 1,
        'GRN' : 2
    }
    accept_drops = True

    path_to_files = [None,None]
    features = None
    default_features = None

    address = None
    port = None

    client_socket = None
    # server_socket = None
    
    is_data_present = False

    data_X = None
    data_Y = None
    are_there_X = False
    are_there_Y = False

    close = False

    def __init__(self):
        super().__init__()

        self.setWindowTitle("GUI prototype")
        # layouts and individual widgets
        layoutP = QVBoxLayout()
        layoutM = QHBoxLayout()
        layoutL = QVBoxLayout()
        layout1 = QHBoxLayout()
        layout2 = QVBoxLayout()
        layout3 = QVBoxLayout()

        # connect widgets to each signal function
        # self.addr_box = AddressBox(self)
        # self.addr_box.signal_addr.connect(self.read_addr)
        # self.port_box = PortBox(self)
        # self.port_box.port_num.connect(self.read_port)
        self.drop_box = DropFiles(self)
        self.drop_box.signal_file_paths.connect(self.read_paths)

        self.graph_canvas = GraphCanvas(self)

        self.feat_form = FeaturesForm(self)
        self.feat_form.signal_check.connect(self.valid_features)
        self.feat_form.disable_all()
        self.feat_form.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)

        # self.default_features = json.loads(open("features_small.json").read())
        self.default_features = {
                                    "WIDTH": [
                                        12.91,
                                        12.9,
                                        12.94
                                    ],
                                    "THICKNESS": [
                                        2.84,
                                        2.88,
                                        2.87
                                    ],
                                    "INTERAXIS": 122.3,
                                    "LT": 0,
                                    "EXTS_LENGTH": 57,
                                    "CS_LENGTH": 50,
                                    "LSF": 0
                                }


        self.cnct_b = ConnectButton(self)
        self.cnct_b.signal_set_socket.connect(self.copy_socket)
        self.disable(self.cnct_b)

        self.upld_b = UploadButton(self)
        self.upld_b.signal_start_conn.connect(self.model_inf)
        self.disable(self.upld_b)
        
        self.stat_check = StatusBar(self)
        self.stat_check.set_status(self.macrostates['RED'])

        self.show_g = ShowGraphs(self, self.graph_canvas)
        self.show_g.signal_get_data.connect(self.pass_data)
        self.disable(self.show_g)

        self.close_b = CloseButton(self)
        self.close_b.signal_end_conn.connect(self.set_end)
        self.disable(self.close_b)

        self.updt_d = UpdateDat(self)
        self.updt_d.signal_update_dat.connect(self.update_dat)
        self.disable(self.updt_d)

        # set widgets to layouts and compose
        #layout1.addWidget(self.addr_box)
        #layout1.addWidget(self.port_box)
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setMovable(False)
        self.tabs.addTab(self.drop_box, "Files")
        self.tabs.addTab(self.graph_canvas, "Graph")
        # self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum )

        self.feat_config = DropMenu(self)
        self.feat_config.signal_set_feats.connect(self.set_features)

        layout2.addWidget(self.tabs)
        widget2 = QWidget()
        widget2.setLayout(layout2)
        
        layout3.addWidget(self.feat_config)
        layout3.addWidget(self.cnct_b)
        layout3.addWidget(self.upld_b)
        layout3.addWidget(self.show_g)
        layout3.addWidget(self.close_b)
        layout3.addWidget(self.updt_d)
        layout3.addWidget(self.feat_form)
        layout3.addWidget(self.stat_check)
        widget3 = QWidget()
        widget3.setLayout(layout3)
        widget3.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

        layoutM.addWidget(widget2)
        layoutM.addWidget(widget3)
        widgetM = QWidget()
        widgetM.setLayout(layoutM)
        
        layoutP.addWidget(widgetM)

        widget = QWidget()
        widget.setLayout(layoutP)

        # finalize
        self.setMinimumSize(QSize(960,540))
        # self.setLayout(layoutP)
        self.setCentralWidget(widget)

        self.state_0()

    def disable(self, widget):
        widget.setEnabled(False)
        widget.setAutoFillBackground(True)
        widget.setStyleSheet("color: #FFFFFF; background: #a90a0a")

    def enable(self, widget):
        widget.setEnabled(True)
        widget.setAutoFillBackground(True)
        widget.setStyleSheet("color: #FFFFFF; background: #59a90a")

    def state(self):
        # check macrostate: 1, 2 or 3
        if self.stat_check.get_status() == self.macrostates['RED']:
            if self.client_socket: # succesfully connected
                self.stat_check.set_status(self.macrostates['YLW'])
            print(f"State: {self.stat_check.get_status()}")
        elif self.stat_check.get_status() == self.macrostates['YLW']:
            if not self.client_socket: # disconnected
                self.stat_check.set_status(self.macrostates['RED'])
            if self.path_to_files[0]: # json path no longer needed
                if self.is_data_present:
                    if self.are_there_X and self.are_there_Y: # succ. inferred w/ the model
                        self.stat_check.set_status(self.macrostates['GRN'])
        elif self.stat_check.get_status() == self.macrostates['GRN']:
            if not self.client_socket: # disconnected
                self.stat_check.set_status(self.macrostates['RED'])

    def state_0(self):
        self.ready_to_send = False
        self.feat_form.misure.setStyleSheet("")
        self.enable(self.cnct_b)
        self.disable(self.upld_b)
        self.disable(self.show_g)
        self.disable(self.close_b)
        self.disable(self.updt_d)
        self.feat_form.disable_all()
        self.drop_box.reset_layout()
        self.graph_canvas.clear_graphs()

    def state_1(self):
        self.disable(self.cnct_b)
        self.enable(self.close_b)
        self.feat_form.enable_all()
        self.drop_box.set_accept(False)
        self.drop_box.setAcceptDrops(True)

    def state_2(self):
        self.feat_form.disable_all()
        self.enable(self.upld_b)

    def state_3(self):
        self.enable(self.show_g)
        self.enable(self.updt_d)
        self.disable(self.upld_b)

    def gray_out(self):
        # disable drop file canvas
        # disable input fields
        # TODO: implement disabling canvas/fields
        self.state()
        if self.client_socket:
            if self.path_to_files[0] :
                self.drop_box.set_accept(True)
                self.drop_box.setAcceptDrops(False)
            else:
                self.state_1()
            if self.is_data_present and self.path_to_files[0]:
                self.state_2()
                if self.are_there_X and self.are_there_Y:
                    self.state_3()
        else:
            self.state_0()

    def switch_tab(self):
        self.tabs.setCurrentIndex(1)

    @pyqtSlot(bool)
    def valid_features(self, cond):
        # check for valid feature fields
        # need wait specification
        # TODO: Implement this
        # if valid features values
        # self.is_data_present = cond

        if cond:
            try:
                self.features = self.feat_form.list_feats()
                print(self.features)
                # self.features["WIDTH"] = self.feat_form.turn_into_array(self.features["WIDTH"])
                # self.features["THICKNESS"] = self.feat_form.turn_into_array(self.features["THICKNESS"])
                # self.features["INTERAXIS"] = float(self.features["INTERAXIS"])
                # self.features["EXTS_LENGTH"] = float(self.features["EXTS_LENGTH"])
                # self.features["CS_LENGTH"] = float(self.features["CS_LENGTH"])
                # self.features["EXTS_ACQUIRED"] = bool(self.features["EXTS_ACQUIRED"])
                self.features["MEASUREMENTS"] = [float(f) for f in self.features["MEASUREMENTS"].split('\t')]
                self.features["WIDTH"] = self.features["MEASUREMENTS"][:3]
                self.features["THICKNESS"] = self.features["MEASUREMENTS"][3:6]
                self.features["LS"] = self.features["MEASUREMENTS"][6]
                # self.features["LT"] = self.features["MEASUREMENTS"][7]
                self.features["LU"] = self.features["MEASUREMENTS"][8]
                self.features["LG"] = self.features["MEASUREMENTS"][9]
                del self.features["MEASUREMENTS"]
                print(self.features)
                self.features = json.dumps(self.features)
                self.feat_form.misure.setStyleSheet("border: 2px solid #59a90a;")
                self.is_data_present = True
                self.gray_out()
                return True
            except Exception as e:
                self.feat_form.misure.setStyleSheet("border: 2px solid red;")
                self.is_data_present = False
                print(e)
                return False
        else:
            return False

    # set default features
    @pyqtSlot(bool)
    def set_features(self):
        self.features = self.default_features

        self.feat_form.set_all(list(self.features.values()))

    # read dropped file paths
    @pyqtSlot(str)
    def read_paths(self, url):
        if os.path.isfile(url):
            print(url)
            self.path_to_files[0] = url
        else:
            # print("Not a file!")
            ErrorBox(self, 7)
        print(self.path_to_files)
        
        self.gray_out()

    # copy socket
    @pyqtSlot(socket.socket)
    def copy_socket(self, socket):
        if socket:
            self.client_socket = socket
        else:
            ErrorBox(self, 3)
        
        self.gray_out()

    @pyqtSlot(bool)
    def pass_data(self, req):
        if req:
            self.show_g.copy_data(self.data_X, self.data_Y)
            self.show_g.update_input()
            self.switch_tab()

    # interact with the model
    @pyqtSlot(bool)
    def model_inf(self, start_sig):
        # no check: it is expected that,
        # would the connect button be clickable,
        # the proper checks have already been passed
        # Atomic Send: all of these funcs at once
        data = self.upld_b.package_data(self.path_to_files[0], self.features)

        self.upld_b.send_data(data) 
        try:
            G_inputs, pred_val = self.upld_b.read_results()
        except Exception as e:
            # ErrorBox(self, 11)
            return 0
        # self.data_X, self.data_Y 
        self.data_X = [
            G_inputs.DISP,
            G_inputs.EXTS,
            pred_val
        ]
        self.data_Y = G_inputs.LOAD
        self.are_there_X = True
        self.are_there_Y = True
        
        self.gray_out()

    @pyqtSlot(bool)
    def update_dat(self, req):
        feat_converted = json.loads(self.features)
        if req:
            feat_formatted = [
                ";".join(feat_converted.keys()),
                ";".join([str(entry) for entry in feat_converted.values()])
            ]
            data_formatted = list(self.data_X[2])
            self.updt_d.write_data_to_file(self.path_to_files[0], feat_formatted, data_formatted)
            self.disable(self.updt_d)

    # end connection and reset gui
    @pyqtSlot(bool)
    def set_end(self, end_sig):
        # no check: it is expected that,
        # would the close button be clickable,
        # the proper checks have alreadu been passed
        # close connections
        # sent EoC custom stream to container
        if self.client_socket:
            data = self.upld_b.embed_header(None)
            self.upld_b.send_data(data)

            self.client_socket = None
        # clear boxes and canvases
        # drop_box.clearAll()
        self.drop_box.remove_icons()
        self.graph_canvas.clear_graphs()
        self.graph_canvas.setParent(None)
        self.graph_canvas = GraphCanvas(self)
        self.tabs.addTab(self.graph_canvas, "Graph")
        self.show_g.set_canvas_item(self.graph_canvas)
        self.feat_form.clear_all()
        self.path_to_files[0]= None
        self.features= None
        self.is_data_present= None
        self.data_X= None
        self.data_Y= None
        self.are_there_X = False
        self.are_there_Y = False
        self.close= False
        self.tabs.setCurrentIndex(0)
        self.drop_box.reset_layout()
        self.gray_out()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    w = MainWindow()
    w.show()

    sys.exit(app.exec())
