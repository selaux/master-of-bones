import os
from PyQt4 import QtCore, QtGui
from helpers.windows.VTKWindow import error_decorator
import skimage
import scipy
import numpy as np
from PIL import ImageQt
from algorithms.segmentation import watershed_cut

def scale_to_max(maximum, pixmap):
    height = min(pixmap.height(), maximum)
    width = min(pixmap.width(), maximum)
    return pixmap.copy(0, 0, pixmap.width(), pixmap.height()).scaled(QtCore.QSize(width, height), QtCore.Qt.KeepAspectRatio)


class ImportBoneWindow(QtGui.QDialog):
    def __init__(self, image_path, image_segmentation_methods):
        QtGui.QDialog.__init__(self)

        self.image_segmentation_methods = image_segmentation_methods

        self.vl = QtGui.QVBoxLayout()

        self._init_segmentation_methods()
        self.raw_skimage = skimage.io.imread(image_path)

        self.raw_image_label = QtGui.QLabel()
        self.raw_image = QtGui.QPixmap(image_path)
        self.raw_image_label.setPixmap(scale_to_max(250, self.raw_image))
        self.vl.addWidget(self.raw_image_label)

        self.il = QtGui.QHBoxLayout()
        self.cut_image_label = QtGui.QLabel()
        self.segmented_image_label = QtGui.QLabel()

        self.il.addWidget(self.cut_image_label)
        self.il.addWidget(self.segmented_image_label)

        self.vl.addLayout(self.il)

        self.labels_box = QtGui.QGroupBox('Select labels marked as on-the-bone (white)')
        self.labels_layout = QtGui.QGridLayout()
        self.labels = []
        self.label_buttons = []
        self.labels_box.setLayout(self.labels_layout)
        self.vl.addWidget(self.labels_box)

        self.setLayout(self.vl)

        bl = QtGui.QHBoxLayout()
        self.ok_button = QtGui.QPushButton('OK')
        self.save_images_button = QtGui.QPushButton('Save Images')
        self.cancel_button = QtGui.QPushButton('Cancel')
        bl.addWidget(self.ok_button)
        bl.addWidget(self.save_images_button)
        bl.addWidget(self.cancel_button)
        self.ok_button.clicked.connect(self.do_accept)
        self.save_images_button.clicked.connect(self.save_images)
        self.cancel_button.clicked.connect(self.do_cancel)

        self.vl.addLayout(bl)

        self.apply_segmentation_method()

    def _init_segmentation_methods(self):
        layout = QtGui.QVBoxLayout()
        self.segmentation_methods_box = QtGui.QGroupBox('Image Segmentation Method')
        self.image_segmentation_buttons = map(lambda e: QtGui.QRadioButton(e['label']), self.image_segmentation_methods)
        for b in self.image_segmentation_buttons:
            b.clicked.connect(self.changed_segmentation_method)

        self.image_segmentation_buttons[0].setChecked(True)
        for b in self.image_segmentation_buttons:
            layout.addWidget(b)
        self.segmentation_methods_box.setLayout(layout)
        self.vl.addWidget(self.segmentation_methods_box)

    @error_decorator
    def apply_segmentation_method(self):
        segmentation_method = list([e for i, e in enumerate(self.image_segmentation_methods) if self.image_segmentation_buttons[i].isChecked()])[0]['fn']
        
        self.raw_cut_image = watershed_cut(self.raw_skimage)

        self.cut_qimage = ImageQt.ImageQt(scipy.misc.toimage(self.raw_cut_image))
        self.cut_image = QtGui.QPixmap.fromImage(self.cut_qimage)
        self.cut_image_label.setPixmap(scale_to_max(400, self.cut_image))

        self.raw_segmented_image = segmentation_method(self.raw_cut_image)
        self.colored_labels_image = ImageQt.ImageQt(scipy.misc.toimage(skimage.color.label2rgb(self.raw_segmented_image)))
        self.segmented_image = QtGui.QPixmap.fromImage(self.colored_labels_image)
        self.segmented_image_label.setPixmap(scale_to_max(400, self.segmented_image))

        self.labels = list(np.unique(self.raw_segmented_image))
        print(len(self.labels))
        for button in self.label_buttons:
            button.close()
        self.label_buttons = map(lambda c: QtGui.QCheckBox(str(int(c))), self.labels)

        col = 0
        row = 0
        if len(self.label_buttons) <= 20:
            for button in self.label_buttons:
                button.clicked.connect(self.changed_classes)
                self.labels_layout.addWidget(button, row, col)
                if (col+1) % 5 == 0:
                    row += 1
                    col = 0
                else:
                    col += 1

    def changed_segmentation_method(self):
        self.apply_segmentation_method()

    def changed_classes(self):
        segmented_and_annotated = skimage.color.label2rgb(self.raw_segmented_image)
        for i, cls in enumerate(self.labels):
            if self.label_buttons[i].isChecked():
                index_array = np.repeat((self.raw_segmented_image == cls).reshape((self.raw_segmented_image.shape[0], self.raw_segmented_image.shape[1], 1)), [3], axis=2)
                segmented_and_annotated[index_array] = 1.0
        self.segmented_image = QtGui.QPixmap.fromImage(ImageQt.ImageQt(scipy.misc.toimage(segmented_and_annotated)))
        self.segmented_image_label.setPixmap(scale_to_max(400, self.segmented_image))

    def do_accept(self):
        if len([b for b in self.label_buttons if b.isChecked()]) == 0:
            QtGui.QMessageBox.critical(
                self,
                'An Error Occured',
                'You have to select at least one label'
            )
            return

        self.bone_pixels = np.zeros_like(self.raw_segmented_image)
        for i, cls in enumerate(self.labels):
            if self.label_buttons[i].isChecked():
                self.bone_pixels[self.raw_segmented_image == cls] = 1.0

        self.accept()

    @error_decorator
    def save_images(self):
        def save_image(path, pixmap):
            pixmap.save(path, os.path.splitext(path)[1][1:])
        dir_name = bytes(QtGui.QFileDialog.getExistingDirectory(
            None,
            caption='Open File to Import',
            directory=os.getcwd()
        ))

        if len(dir_name) > 0:
            save_image(os.path.join(dir_name, 'original.jpg'), self.raw_image)
            save_image(os.path.join(dir_name, 'cut.jpg'), self.cut_image)
            save_image(os.path.join(dir_name, 'segmented.jpg'), self.segmented_image)



    def do_cancel(self):
        self.bone_pixels = None
        self.reject()



