from math import degrees, radians
import vtk
import numpy as np
from PyQt4 import QtCore, QtGui
import traceback
from VTKWindow import VTKWindow
from algorithms.splines import evaluate_spline
from helpers import geometry as gh
from helpers import features as fh
import matplotlib.cm as cmx
from helpers.to_vtk import get_line_actor


class ComparisonWindow(VTKWindow):
    def __init__(self, bones, compare_fn, window_space_functions, feature_functions):
        VTKWindow.__init__(self, title='Comparison Helper')

        self.MIN_WINDOW_SIZE = 0.1
        self.MAX_WINDOW_SIZE = 1.5
        self.separability_properties = [
            {
                'label': 'Precision * Margin',
                'property': 'rm',
            },
            {
                'label': 'Margin',
                'property': 'margin',
            },
            {
                'label': 'Precision',
                'property': 'recall',
            }
        ]

        self.bones = bones
        self.results = None
        self.compare_fn = compare_fn
        self.window_space_functions = window_space_functions
        self.feature_functions = feature_functions

        self.fl_frame = QtGui.QFrame()
        self.fl_frame.setFixedWidth(400)
        self.fl = QtGui.QVBoxLayout()
        self.fl_frame.setLayout(self.fl)
        self.hl.addWidget(self.fl_frame)

        self.window_widget = QtGui.QFrame()
        self.window_widget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.window_vtk_widget, self.window_renderer, self.window_interactor, self.window_istyle = self.init_vtk_widget(self.window_widget)
        self.window_vtk_widget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.window_actors = None
        self.fl.insertWidget(-1, self.window_widget)

        # self.feature_widget = QtGui.QFrame()
        # self.feature_widget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        # self.feature_vtk_widget, self.feature_renderer, self.feature_interactor, self.feature_istyle = self.init_vtk_widget(self.feature_widget)
        # self.fl.insertWidget(-1, self.feature_widget)

        self.init_view_properties()
        self.init_window_extractions()
        self.init_feature_functions()
        self.init_algorithm_parameters()
        self.calc_button = QtGui.QPushButton('Calculate')
        self.fl.addWidget(self.calc_button)
        self.progress_bar = QtGui.QProgressBar()
        self.fl.addWidget(self.progress_bar)

        self.comparison_data = vtk.vtkPolyData()
        self.comparison_mapper = vtk.vtkPolyDataMapper()
        self.comparison_actor = vtk.vtkActor()
        self.detail_angle_data = vtk.vtkPolyData()
        self.detail_angle_mapper = vtk.vtkPolyDataMapper()
        self.detail_angle_actor = vtk.vtkActor()
        self.initialize_actors()
        self.render_actors([self.comparison_actor, self.detail_angle_actor])

        self.window_renderer.AddActor(self.init_scale())

        self.window_interactor.Initialize()
        # self.feature_interactor.Initialize()

        self.iren.AddObserver('LeftButtonPressEvent', self.on_click, 0.0)
        self.calc_button.clicked.connect(self.calculate)

    def init_window_extractions(self):
        layout = QtGui.QVBoxLayout()
        self.window_extraction_box = QtGui.QGroupBox('Window Extraction Type')
        self.window_extraction_buttons = map(lambda e: QtGui.QRadioButton(e['label']), self.window_space_functions)
        self.window_extraction_buttons[0].setChecked(True)
        for b in self.window_extraction_buttons:
            layout.addWidget(b)
        self.window_extraction_box.setLayout(layout)
        self.fl.addWidget(self.window_extraction_box)

    def init_feature_functions(self):
        layout = QtGui.QVBoxLayout()
        self.feature_fn_box = QtGui.QGroupBox('Features')
        self.feature_fn_buttons = map(lambda e: QtGui.QRadioButton(e['label']), self.feature_functions)
        self.feature_fn_buttons[0].setChecked(True)
        for b in self.feature_fn_buttons:
            layout.addWidget(b)
        self.feature_fn_box.setLayout(layout)
        self.fl.addWidget(self.feature_fn_box)

    def init_view_properties(self):
        layout = QtGui.QFormLayout()
        self.show_metric_box = QtGui.QComboBox()
        for e in self.separability_properties:
            self.show_metric_box.addItem(e['label'])
        self.show_metric_box.currentIndexChanged.connect(self.update_overview_data)
        self.show_metric_box.setCurrentIndex(0)
        layout.addRow('Shown Metric', self.show_metric_box)

        self.mean_bone_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.mean_bone_slider.setMinimum(0)
        self.mean_bone_slider.setMaximum(100)
        self.mean_bone_slider.setValue(50)
        self.mean_bone_slider.valueChanged.connect(self.update_overview_data)
        layout.addRow('Shift Mean Bone', self.mean_bone_slider)

        self.vl.addLayout(layout)


    def init_algorithm_parameters(self):
        layout = QtGui.QFormLayout()
        self.angle_spinbox = QtGui.QSpinBox()
        self.angle_spinbox.setValue(5)
        layout.addRow('Evaluate every x degrees', self.angle_spinbox)

        self.window_size_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.window_size_slider.setMinimum(0)
        self.window_size_slider.setMaximum(100)
        self.window_size_slider.setValue(50)
        layout.addRow('Window Size', self.window_size_slider)

        self.number_of_spline_evaluations_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.number_of_spline_evaluations_slider.setMinimum(3)
        self.number_of_spline_evaluations_slider.setMaximum(100)
        self.number_of_spline_evaluations_slider.setValue(25)
        layout.addRow('Number of Spline Evaluations', self.number_of_spline_evaluations_slider)

        self.use_pca_checkbox = QtGui.QCheckBox()
        self.use_pca_checkbox.setChecked(True)
        layout.addRow('Use PCA', self.use_pca_checkbox)

        self.number_of_pca_components_spinbox = QtGui.QSpinBox()
        self.number_of_pca_components_spinbox.setValue(4)
        layout.addRow('Number of PCA Components', self.number_of_pca_components_spinbox)

        self.fl.addLayout(layout)

    def initialize_actors(self):
        self.comparison_mapper.SetInputData(self.comparison_data)
        self.comparison_actor.SetMapper(self.comparison_mapper)
        self.comparison_actor.GetProperty().SetRepresentationToWireframe()
        self.comparison_actor.GetProperty().SetLineWidth(3)

        self.detail_angle_mapper.SetInputData(self.detail_angle_data)
        self.detail_angle_actor.SetMapper(self.detail_angle_mapper)
        self.detail_angle_actor.GetProperty().SetRepresentationToWireframe()
        self.detail_angle_actor.GetProperty().SetLineWidth(1.5)
        self.detail_angle_actor.GetProperty().SetColor(0, 0, 0)


    def get_window_size(self):
        return self.MIN_WINDOW_SIZE + self.MAX_WINDOW_SIZE * self.window_size_slider.value() / float(self.window_size_slider.maximum())

    def calculate(self):
        try:
            self.calc_button.setEnabled(False)
            QtGui.QApplication.processEvents()

            feature_fn = list([f for i, f in enumerate(self.feature_functions) if self.feature_fn_buttons[i].isChecked()])[0]['fn']
            extract_window_fn = list([e for i, e in enumerate(self.window_space_functions) if self.window_extraction_buttons[i].isChecked()])[0]['fn']
            window_size = self.get_window_size()
            use_pca = self.use_pca_checkbox.isChecked()
            number_of_pca_components = self.number_of_pca_components_spinbox.value()
            number_of_spline_evaluations = self.number_of_spline_evaluations_slider.value()
            step_size = self.angle_spinbox.value()

            kwargs = {
                'feature_fn': feature_fn,
                'extract_window_fn': extract_window_fn,
                'step_size': step_size,
                'progress_callback': self.update_progress_bar,
                'window_size': window_size,
                'number_of_evaluations': number_of_spline_evaluations,
                'use_pca': use_pca,
                'pca_components': number_of_pca_components
            }

            self.results = self.compare_fn(self.bones, 2, 3, **kwargs)

            QtGui.QApplication.processEvents()
            self.calc_button.setEnabled(True)

            self.update_overview_data()
            self.update_detailed_data(90)
        except:
            print(traceback.format_exc())

    def update_progress_bar(self, progress, max_progress):
        self.progress_bar.setMaximum(max_progress)
        self.progress_bar.setValue(progress)
        QtGui.QApplication.processEvents()

    def get_current_mean_outline(self, class1, class2):
        space = np.linspace(0, 1, 250)

        class1bones = np.array([evaluate_spline(space, o['spline_params']) for o in self.bones if o['class'] == class1])
        class1part = self.mean_bone_slider.value() / 100.0
        class2bones = np.array([evaluate_spline(space, o['spline_params']) for o in self.bones if o['class'] == class2])
        class2part = 1 - class1part

        return np.mean(class1bones, axis=0) * class1part + np.mean(class2bones, axis=0) * class2part

    def update_overview_data(self):
        try:
            if self.results:
                points = vtk.vtkPoints()
                colors = vtk.vtkUnsignedCharArray()
                colors.SetNumberOfComponents(3)
                colors.SetName("Colors")
                vertices = vtk.vtkCellArray()
                lines = vtk.vtkCellArray()
                mean_outline = self.get_current_mean_outline(2, 3)
                angles = np.array([m['angle'] for m in self.results])
                shown_property = self.separability_properties[self.show_metric_box.currentIndex()]['property']
                print(shown_property)
                separabilities = fh.normalize(np.array([ m[shown_property] for m in self.results ]))

                for i, point in enumerate(mean_outline):
                    points.InsertNextPoint([point[1], point[0], 1.0])
                    vertex = vtk.vtkVertex()
                    vertex.GetPointIds().SetId(0, i)
                    vertices.InsertNextCell(vertex)

                    rho, phi = gh.cart2pol(point[0], point[1])
                    phi = degrees(phi) + 360 if phi < 0 else degrees(phi)
                    closest = np.argmin(np.abs(angles - phi))
                    separability = separabilities[closest]
                    raw_color = cmx.gnuplot(separability)
                    color = (int(raw_color[0] * 255), int(raw_color[1] * 255), int(raw_color[2] * 255))
                    colors.InsertNextTuple3(*color)

                for i in range(0, mean_outline.shape[0]):
                    j = i+1 if i+1 < mean_outline.shape[0] else 0
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, i)
                    line.GetPointIds().SetId(1, j)
                    lines.InsertNextCell(line)

                self.comparison_data.SetPoints(points)
                self.comparison_data.SetLines(lines)
                self.comparison_data.SetVerts(vertices)
                self.comparison_data.GetPointData().SetScalars(colors)
                self.comparison_data.Modified()

                self.ren.ResetCamera()
                self.vtkWidget.GetRenderWindow().Render()
        except:
            print(traceback.format_exc())

    def update_detailed_data(self, angle):
        try:
            angles = np.array([m['angle'] for m in self.results])
            closest = np.argmin(np.abs(angles - angle))
            evaluation = self.results[closest]
            # current_angle = angles[closest]
            # y, x = gh.pol2cart(2, radians(current_angle))
            #
            # points = vtk.vtkPoints()
            # points.InsertNextPoint([0, 0, 1.0])
            # points.InsertNextPoint([x, y, 1.0])
            # lines = vtk.vtkCellArray()
            # line = vtk.vtkLine()
            # line.GetPointIds().SetId(0, 0)
            # line.GetPointIds().SetId(1, 1)
            # lines.InsertNextCell(line)
            # self.detail_angle_data.SetPoints(points)
            # self.detail_angle_data.SetLines(lines)
            # self.detail_angle_data.Modified()

            if self.window_actors:
                for actor in self.window_actors:
                    self.window_renderer.RemoveActor(actor)

            self.window_actors = []
            for i, bone in enumerate(self.bones):
                spline_params = bone['spline_params']
                window = evaluation['windows'][i, :]
                points = evaluate_spline(window, spline_params)
                features = evaluation['features'][i, :]

                actor = get_line_actor(points)
                actor.GetProperty().SetRepresentationToWireframe()
                actor.GetProperty().SetLineWidth(1.5)
                actor.GetProperty().SetColor(bone['color'][0], bone['color'][1], bone['color'][2])

                self.window_actors.append(actor)
                self.window_renderer.AddActor(actor)

            self.window_renderer.ResetCamera()
            self.vtkWidget.GetRenderWindow().Render()
            self.window_vtk_widget.GetRenderWindow().Render()
        except:
            print(traceback.format_exc())

    def on_click(self, obj, event):
        if self.results:
            mouse_position = obj.GetEventPosition()
            if mouse_position:
                self.ren.SetDisplayPoint((mouse_position[0], mouse_position[1], 0))
                self.ren.DisplayToWorld()
                x, y, z, w = self.ren.GetWorldPoint()

                rho, phi = gh.cart2pol(y, x)
                phi = degrees(phi) + 360 if phi < 0 else degrees(phi)

                self.update_detailed_data(phi)





