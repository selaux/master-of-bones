from functools import partial
import traceback
from PyQt4 import QtGui, QtCore
import numpy as np
from VTKWindow import VTKWindow
from algorithms.generation import generate_ellipse
from helpers.to_vtk import get_outline_actor


class SyntheticGenerationWindow(VTKWindow):
    def __init__(self):
        VTKWindow.__init__(self, title='Generate Synthetic Data')

        self.bones = []

        self.fl_frame = QtGui.QFrame()
        self.fl_frame.setFixedWidth(600)
        self.fl = QtGui.QVBoxLayout()
        self.fl_frame.setLayout(self.fl)
        self.hl.addWidget(self.fl_frame)

        self.class1_form = QtGui.QGroupBox('Class 1')
        self.class2_form = QtGui.QGroupBox('Class 2')
        self.class1_form.setLayout(self.init_algorithm_parameters(self.class1_form))
        self.class2_form.setLayout(self.init_algorithm_parameters(self.class2_form))
        self.fl.addWidget(self.class1_form)
        self.fl.addWidget(self.class2_form)

        self.generate_button = QtGui.QPushButton('Generate Models')
        self.fl.addWidget(self.generate_button)

        self.actors = []
        self.render_actors(self.actors)

        self.generate_button.clicked.connect(self.generate)

    def init_algorithm_parameters(self, clsObj):
        def add_point():
            number_of_items = len(clsObj.table_model.special_points)
            clsObj.table_model.beginInsertRows(QtCore.QModelIndex(), number_of_items, number_of_items)
            clsObj.table_model.special_points.append({
                'angle': '45',
                'angle_std_dev': '5',
                'height': '0.1',
                'height_std_dev': '0.001',
                'std_dev': '5',
                'std_dev_std_dev': '0.1'
            })
            clsObj.table_model.endInsertRows()

        def remove_point():
            indices = map(lambda i: i.row(), clsObj.points_table.selectionModel().selectedRows())
            for i in indices:
                clsObj.table_model.beginRemoveRows(QtCore.QModelIndex(), i, i)
                del clsObj.table_model.special_points[i]
                clsObj.table_model.endRemoveRows()


        layout = QtGui.QFormLayout()

        clsObj.number_of_observations = QtGui.QSpinBox()
        clsObj.number_of_observations.setValue(25)
        layout.addRow('Number of Observations', clsObj.number_of_observations)

        clsObj.max_points_layout = self.init_slider_param(25, 500, 250)
        layout.addRow('Max Number of Points', clsObj.max_points_layout)

        clsObj.min_points_layout = self.init_slider_param(25, 500, 150)
        layout.addRow('Min Number of Points', clsObj.min_points_layout)

        clsObj.eccentricity_layout = self.init_slider_param(0, 100, 0, lambda x: 1.0 + x*3.0/100.0)
        layout.addRow('Eccentricity', clsObj.eccentricity_layout)

        clsObj.noise_angle_layout = self.init_slider_param(0, 100, 0, lambda x: x/100.0)
        layout.addRow('Standard Deviation Angle', clsObj.noise_angle_layout)

        clsObj.noise_radius_layout = self.init_slider_param(0, 100, 0, lambda x: x*0.05/100.0)
        layout.addRow('Standard Deviation Radius', clsObj.noise_radius_layout)

        clsObj.noise_transformation_layout = self.init_slider_param(0, 100, 0, lambda x: x*0.05/100.0)
        layout.addRow('Standard Deviation Transformation', clsObj.noise_transformation_layout)

        clsObj.points_table = QtGui.QTableView()
        clsObj.table_model = SpecialPointsTableModel(self)
        clsObj.points_table.setModel(clsObj.table_model)
        layout.addRow(QtGui.QLabel('Special Points'))

        clsObj.add_button = QtGui.QPushButton('Add')
        clsObj.remove_button = QtGui.QPushButton('Remove')
        clsObj.btn_layout = QtGui.QHBoxLayout()
        clsObj.btn_layout.addWidget(clsObj.add_button)
        clsObj.btn_layout.addWidget(clsObj.remove_button)
        clsObj.add_button.clicked.connect(add_point)
        clsObj.remove_button.clicked.connect(remove_point)
        layout.addRow(clsObj.btn_layout)

        layout.addRow(clsObj.points_table)

        return layout

    def init_slider_param(self, min, max, init, mapping=lambda x: x):
        layout = QtGui.QHBoxLayout()
        layout.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        layout.slider.setMinimum(min)
        layout.slider.setMaximum(max)
        layout.slider.setValue(init)
        layout.addWidget(layout.slider)
        layout.label = QtGui.QLabel()
        layout.label.setFixedWidth(40)
        layout.addWidget(layout.label)

        layout.slider.mapped_value = lambda: mapping(layout.slider.value())

        def set_label():
            layout.label.setText(str(layout.slider.mapped_value()))

        layout.slider.valueChanged.connect(set_label)
        set_label()

        return layout

    def calculate_class(self, form):
        min_points = form.min_points_layout.slider.mapped_value()
        max_points = form.max_points_layout.slider.mapped_value()
        eccentricity = form.eccentricity_layout.slider.mapped_value()
        std_dev_angle = form.noise_angle_layout.slider.mapped_value()
        std_dev_radius = form.noise_radius_layout.slider.mapped_value()
        std_dev_transformation = form.noise_transformation_layout.slider.mapped_value()
        special_points = map(lambda p: {
            'angle': float(p['angle'].replace(',', '.')),
            'angle_std_dev': float(p['angle_std_dev'].replace(',', '.')),
            'height': float(p['height'].replace(',', '.')),
            'height_std_dev': float(p['height_std_dev'].replace(',', '.')),
            'std_dev': float(p['std_dev'].replace(',', '.')),
            'std_dev_std_dev': float(p['std_dev_std_dev'].replace(',', '.'))
        }, form.table_model.special_points)

        return map(lambda x: generate_ellipse(
            eccentricity,
            min_points,
            max_points,
            std_dev_angle,
            std_dev_radius,
            std_dev_transformation,
            special_points
        ), range(form.number_of_observations.value()))

    def generate(self):
        try:
            self.bones1 = self.calculate_class(self.class1_form)
            self.bones2 = self.calculate_class(self.class2_form)
            self.update_data()
        except:
            print(traceback.format_exc())

    def update_data(self):
        def get_actor(color, outline):
            num_edges = outline.shape[0]
            edges = np.zeros((num_edges, 2), dtype=np.int)
            edges[:, 0] = range(0, num_edges)
            edges[:, 1] = range(1, num_edges+1)
            edges[-1, 1] = 0
            return get_outline_actor({
                'points': outline,
                'edges': edges
            }, color, 0xFFFF, False)

        for actor in self.actors:
            self.ren.RemoveActor(actor)

        self.actors = map(partial(get_actor, (1.0, 0, 0)), self.bones1) + map(partial(get_actor, (0, 1.0, 0)), self.bones2)

        for actor in self.actors:
            self.ren.AddActor(actor)

        self.ren.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

class SpecialPointsTableModel(QtCore.QAbstractTableModel):
    properties_map = {
        0: 'angle',
        1: 'angle_std_dev',
        2: 'height',
        3: 'height_std_dev',
        4: 'std_dev',
        5: 'std_dev_std_dev'
    }

    def __init__(self, parent=None, *args):
        """ datain: a list of lists
            headerdata: a list of strings
        """
        QtCore.QAbstractTableModel.__init__(self, parent, *args)
        self.special_points = []

    def rowCount(self, parent):
        return len(self.special_points)

    def columnCount(self, parent):
        return 6

    def data(self, index, role):
        if not index.isValid():
            return QtCore.QVariant()
        elif role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()
        return QtCore.QVariant(self.special_points[index.row()][self.properties_map[index.column()]])

    def setData(self, index, value, role):
        self.special_points[index.row()][self.properties_map[index.column()]] = value.toPyObject()
        return True

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self.properties_map[col])
        if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(col)
        return QtCore.QVariant()

    def flags(self, index):
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsSelectable
