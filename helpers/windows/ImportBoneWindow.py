from PyQt4 import QtCore, QtGui

class ImportBoneWindow(QtGui.QDialog):
    def __init__(self, base_save_path, image_path, image_segmentation_methods):
        QtGui.QDialog.__init__(self)

        self.base_save_path = base_save_path
        self.image_segmentation_methods = image_segmentation_methods

        self.vl = QtGui.QVBoxLayout()

        self._init_segmentation_methods()

        self.il = QtGui.QHBoxLayout()

        self.raw_image_label = QtGui.QLabel()
        self.raw_image = QtGui.QPixmap(image_path)
        self.raw_image_label.setPixmap(self.raw_image)

        self.vl.addLayout(self.il)

        self.setLayout(self.vl)

    def _init_segmentation_methods(self):
        layout = QtGui.QVBoxLayout()
        self.segmentation_methods_box = QtGui.QGroupBox('Image Segmentation Method')
        self.image_segmentation_buttons = map(lambda e: QtGui.QRadioButton(e['label']), self.image_segmentation_methods)
        self.image_segmentation_buttons[0].setChecked(True)
        for b in self.image_segmentation_buttons:
            layout.addWidget(b)
        self.segmentation_methods_box.setLayout(layout)
        self.vl.addWidget(self.segmentation_methods_box)

    def apply_segmentation_method(self):
        pass

