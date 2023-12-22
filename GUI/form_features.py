import sys
from PyQt6.QtWidgets import (
    QSizePolicy,
    QFormLayout,
    QWidget,
    QLabel,
    QLineEdit
)
from PyQt6.QtCore import pyqtSignal, pyqtSlot
import sys
from main_fonts import (
    FontH,
    FontH_s
)

# TODO: Implement each Features field

class FeaturesForm(QWidget):
    
    form = QFormLayout()
    signal_check = pyqtSignal(bool)

    def __init__(self, parent = None):
        super().__init__(parent)

        self.form.ItemRole(2)

        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        font_1 = FontH_s(FontH())

        # for i in range(7):
        #     self.form.addRow(QLabel(f"Features{i}"), QLineEdit())

        self.filename = QLineEdit()
        
        # self.width = QLineEdit()
        # self.width.returnPressed.connect(self.check_upload)
        # self.width.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        # self.width.setMinimumSize(143,24)

        # self.thickness = QLineEdit()
        # self.thickness.returnPressed.connect(self.check_upload)
        # self.thickness.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        # self.thickness.setMinimumSize(143,24)

        # self.interaxis = QLineEdit()
        # self.interaxis.returnPressed.connect(self.check_upload)
        # self.interaxis.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        # self.interaxis.setMinimumSize(143,24)

        # self.exts_length = QLineEdit()
        # self.exts_length.returnPressed.connect(self.check_upload)
        # self.exts_length.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        # self.exts_length.setMinimumSize(143,24)

        # self.cs_length = QLineEdit()
        # self.cs_length.returnPressed.connect(self.check_upload)
        # self.cs_length.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        # self.cs_length.setMinimumSize(143,24)

        # self.exts_acquired = QLineEdit()
        # self.exts_acquired.returnPressed.connect(self.check_upload)
        # self.exts_acquired.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        # self.exts_acquired.setMinimumSize(143,24)

        self.misure = QLineEdit()
        self.misure.returnPressed.connect(self.check_upload)
        self.misure.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.misure.setMinimumSize(143,24)

        # self.width_label = QLabel("WIDTH")
        # self.width_label.setFont(font_1)
        # self.thickness_label = QLabel("THICKNESS")
        # self.thickness_label.setFont(font_1)
        # self.interaxis_label = QLabel("INTERAXIS")
        # self.interaxis_label.setFont(font_1)
        # self.exts_length_label = QLabel("EXTS_LENGTH")
        # self.exts_length_label.setFont(font_1)
        # self.cs_length_label = QLabel("CS_LENGTH")
        # self.cs_length_label.setFont(font_1)
        self.misure_label = QLabel("MEASUREMENTS")
        self.misure_label.setFont(font_1)
        # self.exts_acquired_label = QLabel("EXTS_ACQUIRED")
        # self.exts_acquired_label.setFont(font_1)

        # self.form.addRow(QLabel("FILENAME"), self.filename )
        # self.form.addRow(self.width_label, self.width )
        # self.form.addRow(self.thickness_label, self.thickness )
        # self.form.addRow(self.interaxis_label, self.interaxis )
        # self.form.addRow(self.exts_length_label, self.exts_length )
        # self.form.addRow(self.cs_length_label, self.cs_length )
        # self.form.addRow(self.exts_acquired_label, self.exts_acquired )
        self.form.addRow(self.misure_label, self.misure)

        self.setLayout(self.form)

        self.setSizePolicy(QSizePolicy.Policy.Minimum,QSizePolicy.Policy.Minimum )

        # self.returnPressed.connect(self.check_upload)

    def turn_into_array(self, str = None):
        # [x.xxx,y.yyy,z.zzz]
        # return str[1:-1].strip().split(',')
        res = list(str.strip().split(','))
        return [float(x) for x in res]

    def list_feats(self):
        features = {}
        index, value = [], []
        for i in range(self.layout().count()):
            # optimize with itemAt(i+1)
            row = self.form.itemAt(i)

            if (i % 2 == 0):
                index.append(row.widget().text())
            else:
                value.append(row.widget().text().strip("[]"))
        for (i, feature) in enumerate(index):
            features[feature] = value[i]
        return features

    def disable_all(self):
        self.filename.setReadOnly(True)
        # self.width.setReadOnly(True)
        # self.thickness.setReadOnly(True)
        # self.interaxis.setReadOnly(True)
        # self.exts_length.setReadOnly(True)
        # self.cs_length.setReadOnly(True)
        # self.exts_acquired.setReadOnly(True)
        self.misure.setReadOnly(True)

    def enable_all(self):
        self.filename.setReadOnly(False)
        # self.width.setReadOnly(False)
        # self.thickness.setReadOnly(False)
        # self.interaxis.setReadOnly(False)
        # self.exts_length.setReadOnly(False)
        # self.cs_length.setReadOnly(False)
        # self.exts_acquired.setReadOnly(False)
        self.misure.setReadOnly(False)

    def clear_all(self):
        self.filename.clear()
        # self.width.clear()
        # self.thickness.clear()
        # self.interaxis.clear()
        # self.exts_length.clear()
        # self.cs_length.clear()
        # self.exts_acquired.clear()
        self.misure.clear()

    def format_iterable_obj(self, iterable_obj):
        misure_list = []
        for its in iterable_obj:
            if type(its) == list:
                for it in its:
                    misure_list.append(str(it))
            else:
                misure_list.append(str(its))
        return '\t'.join(misure_list)

    def set_all(self, iterable_obj):
        # self.width.setText(str(iterable_obj[0]))
        # self.thickness.setText(str(iterable_obj[1]))
        # self.interaxis.setText(str(iterable_obj[2]))
        # self.exts_length.setText(str(iterable_obj[3]))
        # self.cs_length.setText(str(iterable_obj[4]))
        misure_list = self.format_iterable_obj(iterable_obj)
        self.misure.setText(misure_list)

    @pyqtSlot()
    def check_upload(self):
        self.signal_check.emit(True)

