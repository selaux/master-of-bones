from math import degrees
import vtk
from PyQt4 import QtCore, QtGui
from VTKWindow import VTKWindow, error_decorator
from helpers import geometry as gh
from helpers import classes as ch

class ComparisonWindow(VTKWindow):
    def __init__(self, bones, compare_fn, window_extractors, feature_extractors):
        VTKWindow.__init__(self, title='Comparison Helper')

        self.MIN_WINDOW_SIZE = 0.1
        self.MAX_WINDOW_SIZE = 1

        self.bones = bones
        self.results = None
        self.compare_fn = compare_fn
        self.window_extractors = window_extractors
        self.feature_extractors = feature_extractors

        self.classes_in_data = list(set(map(lambda b: b['class'], self.bones)))

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
        self.save_main_vizualization_button = self.init_save_vizualization_button(
            self.window_widget,
            self.window_vtk_widget.GetRenderWindow()
        )

        self.graph_widget = QtGui.QFrame()
        self.graph_widget.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.graph_vtk_widget, self.graph_renderer, self.graph_interactor, self.graph_istyle = self.init_vtk_widget(self.graph_widget)
        self.graph_vtk_widget.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.graph_context = vtk.vtkContextView()
        self.graph_context.SetRenderWindow(self.graph_vtk_widget.GetRenderWindow())
        self.graph_views = None
        self.vl.addWidget(self.graph_widget)
        self.save_main_vizualization_button = self.init_save_vizualization_button(
            self.graph_widget,
            self.graph_vtk_widget.GetRenderWindow()
        )
        self.graph_item = None

        # Fixme: Implement Feature Widget (shows features of current window)
        # self.feature_widget = QtGui.QFrame()
        # self.feature_widget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        # self.feature_vtk_widget, self.feature_renderer, self.feature_interactor, self.feature_istyle = self.init_vtk_widget(self.feature_widget)
        # self.fl.insertWidget(-1, self.feature_widget)

        self.init_compared_classes()
        self.init_window_extractions()
        self.init_feature_functions()
        self.init_algorithm_parameters()
        self.init_view_properties()

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
        self.graph_interactor.Initialize()
        # self.feature_interactor.Initialize()

        self.iren.AddObserver('LeftButtonPressEvent', self.on_click, 0.0)
        self.calc_button.clicked.connect(self.calculate)

    def init_compared_classes(self):
        self.classes_box = QtGui.QGroupBox('Select Compared Classes')
        self.classes_checkboxes = []
        classes_layout = QtGui.QVBoxLayout()
        for i, c in enumerate(self.classes_in_data):
            b = QtGui.QCheckBox(ch.get_class_name(c))
            if i < 2:
                b.setChecked(True)
            self.classes_checkboxes.append(b)
            classes_layout.addWidget(b)
        self.classes_box.setLayout(classes_layout)
        self.fl.addWidget(self.classes_box)

    def init_window_extractions(self):
        layout = QtGui.QVBoxLayout()
        self.window_extraction_box = QtGui.QGroupBox('Window Extraction Type')
        self.window_extraction_buttons = map(lambda e: QtGui.QRadioButton(e['label']), self.window_extractors)
        self.window_extraction_buttons[0].setChecked(True)
        for b in self.window_extraction_buttons:
            layout.addWidget(b)
        self.window_extraction_box.setLayout(layout)
        self.fl.addWidget(self.window_extraction_box)

    def init_feature_functions(self):
        layout = QtGui.QVBoxLayout()
        self.feature_fn_box = QtGui.QGroupBox('Features')
        self.feature_fn_buttons = map(lambda e: QtGui.QRadioButton(e['label']), self.feature_extractors)
        self.feature_fn_buttons[0].setChecked(True)
        for b in self.feature_fn_buttons:
            layout.addWidget(b)
        self.feature_fn_box.setLayout(layout)
        self.fl.addWidget(self.feature_fn_box)

    def init_view_properties(self):
        layout = QtGui.QFormLayout()

        self.max_separability_metric_label = QtGui.QLabel()
        layout.addRow('Maximum Value', self.max_separability_metric_label)

        self.min_separability_metric_label = QtGui.QLabel()
        layout.addRow('Minimum Value', self.min_separability_metric_label)

        self.show_metric_box = QtGui.QComboBox()
        self.show_metric_box.currentIndexChanged.connect(self.update_overview_data)
        self.show_metric_box.setCurrentIndex(0)
        layout.addRow('Shown Metric', self.show_metric_box)

        mbs_layout = QtGui.QHBoxLayout()
        class1 = self.get_compared_classes()[1]
        num_class1 = len(filter(lambda b: b['class'] == class1, self.bones))
        mbs_layout.addWidget(QtGui.QLabel('{} ({})'.format(ch.get_class_name(class1), num_class1)))
        self.mean_bone_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.mean_bone_slider.setMinimum(0)
        self.mean_bone_slider.setMaximum(100)
        self.mean_bone_slider.setValue(50)
        self.mean_bone_slider.valueChanged.connect(self.update_overview_data)
        mbs_layout.addWidget(self.mean_bone_slider)
        class2 = self.get_compared_classes()[0]
        num_class2 = len(filter(lambda b: b['class'] == class2, self.bones))
        mbs_layout.addWidget(QtGui.QLabel('{} ({})'.format(ch.get_class_name(class2), num_class2)))
        layout.addRow('Shift Mean Bone', mbs_layout)

        self.vl.addLayout(layout)

    def init_algorithm_parameters(self):
        layout = QtGui.QFormLayout()

        self.angle_spinbox = QtGui.QSpinBox()
        self.angle_spinbox.setValue(5)
        layout.addRow('Evaluate every x degrees', self.angle_spinbox)

        window_size_layout = QtGui.QHBoxLayout()
        self.window_size_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.window_size_slider.setMinimum(0)
        self.window_size_slider.setMaximum(100)
        self.window_size_slider.setValue(50)
        window_size_layout.addWidget(self.window_size_slider)
        self.window_size_label = QtGui.QLabel()
        self.window_size_label.setFixedWidth(30)
        window_size_layout.addWidget(self.window_size_label)
        def set_label_window_size():
            self.window_size_label.setText(str(self.get_window_size()))
        self.window_size_slider.valueChanged.connect(set_label_window_size)
        layout.addRow('Window Size', window_size_layout)
        set_label_window_size()

        spline_eval_layout = QtGui.QHBoxLayout()
        self.number_of_spline_evaluations_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.number_of_spline_evaluations_slider.setMinimum(3)
        self.number_of_spline_evaluations_slider.setMaximum(100)
        self.number_of_spline_evaluations_slider.setValue(25)
        spline_eval_layout.addWidget(self.number_of_spline_evaluations_slider)
        self.spline_evals_label = QtGui.QLabel()
        self.spline_evals_label.setFixedWidth(30)
        spline_eval_layout.addWidget(self.spline_evals_label)
        def set_label_spline():
            self.spline_evals_label.setText(str(self.number_of_spline_evaluations_slider.value()))
        self.number_of_spline_evaluations_slider.valueChanged.connect(set_label_spline)
        set_label_spline()

        layout.addRow('Number of Spline Evaluations', spline_eval_layout)

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

    def get_compared_classes(self):
        classes = list([c for i, c in enumerate(self.classes_in_data) if self.classes_checkboxes[i].isChecked()])
        if len(classes) != 2:
            QtGui.QMessageBox.information(
                self,
                'Info',
                'You need to check exactly two classes to compare. Falling back to using defaults.'
            )
            return self.classes_in_data[:2]
        return classes

    def get_window_size(self):
        return self.MIN_WINDOW_SIZE + self.MAX_WINDOW_SIZE * self.window_size_slider.value() / float(self.window_size_slider.maximum())

    @error_decorator
    def calculate(self):
        self.calc_button.setEnabled(False)
        QtGui.QApplication.processEvents()

        feature_extractor = list([f for i, f in enumerate(self.feature_extractors) if self.feature_fn_buttons[i].isChecked()])[0]['fn']
        window_extractor = list([e for i, e in enumerate(self.window_extractors) if self.window_extraction_buttons[i].isChecked()])[0]['fn']
        window_size = self.get_window_size()
        use_pca = self.use_pca_checkbox.isChecked()
        number_of_pca_components = self.number_of_pca_components_spinbox.value()
        number_of_spline_evaluations = self.number_of_spline_evaluations_slider.value()
        step_size = self.angle_spinbox.value()
        class1, class2 = self.get_compared_classes()

        kwargs = {
            'feature_extractor': feature_extractor,
            'window_extractor': window_extractor,
            'step_size': step_size,
            'progress_callback': self.update_progress_bar,
            'window_size': window_size,
            'number_of_evaluations': number_of_spline_evaluations,
            'use_pca': use_pca,
            'pca_components': number_of_pca_components
        }

        self.results = self.compare_fn(self.bones, class1, class2, **kwargs)

        QtGui.QApplication.processEvents()
        self.calc_button.setEnabled(True)

        self.reset_overview()
        self.update_overview_data()
        self.update_detailed_data(90)

    def update_progress_bar(self, progress, max_progress):
        self.progress_bar.setMaximum(max_progress)
        self.progress_bar.setValue(progress)
        QtGui.QApplication.processEvents()

    @error_decorator
    def reset_overview(self):
        if self.results:
            old_metric_index = self.show_metric_box.currentIndex()
            old_metric_index = old_metric_index if old_metric_index != -1 else 0
            self.show_metric_box.clear()
            for m in self.results.single_results[0].get_performance_indicators():
                self.show_metric_box.addItem(m['label'])
            self.show_metric_box.setCurrentIndex(old_metric_index)

            if self.comparison_actor:
                self.ren.RemoveActor(self.comparison_actor)
            if self.graph_item:
                self.graph_context.GetScene().RemoveItem(self.graph_item)
            self.comparison_actor = self.results.actor
            self.graph_item = self.results.chart
            self.ren.AddActor(self.comparison_actor)
            self.graph_context.GetScene().AddItem(self.graph_item)
            self.update_overview_data()

            self.ren.ResetCamera()
            self.vtkWidget.GetRenderWindow().Render()

    @error_decorator
    def update_overview_data(self):
        if self.results:
            ratio = self.mean_bone_slider.value() / 100.0
            performance_indicator_index = self.show_metric_box.currentIndex()
            min_indicator, max_indicator = self.results.get_min_and_max_performance_indicators(performance_indicator_index)

            self.min_separability_metric_label.setText(str(min_indicator))
            self.max_separability_metric_label.setText(str(max_indicator))

            self.results.update_actor(ratio, performance_indicator_index)

            self.vtkWidget.GetRenderWindow().Render()

    @error_decorator
    def update_detailed_data(self, angle):
        single_result = self.results.get_closest_single_result(angle)

        if self.window_actors:
            for actor in self.window_actors:
                self.window_renderer.RemoveActor(actor)

        self.window_actors = single_result.get_windows_actors()
        for actor in self.window_actors:
            self.window_renderer.AddActor(actor)

        self.window_renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
        self.window_vtk_widget.GetRenderWindow().Render()

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





