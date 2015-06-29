from functools import partial
import traceback
import numpy as np
from VTKWindow import VTKWindow, error_decorator
from PyQt4 import QtGui
import os
from ..to_vtk import get_outline_actor
import helpers.loading as lh


class RegistrationWindow(VTKWindow):
    def __init__(self, bones, register_fn, estimators, reference_estimators):
        VTKWindow.__init__(self, title='Registration Helper')

        self.bones = bones
        self.register_fn = register_fn
        self.estimators = estimators
        self.reference_estimators = reference_estimators
        self.initial_registration_done = False

        self.initialize_actors()

        self.sl = QtGui.QVBoxLayout()
        self.init_outline_checkboxes()
        self.init_class_selector()

        self.progress_bar = QtGui.QProgressBar()
        self.sl.insertWidget(0, self.progress_bar)

        self.calc_button = QtGui.QPushButton('Calculate')
        self.sl.insertWidget(0, self.calc_button)

        self.al = QtGui.QHBoxLayout()
        self.init_estimators()
        self.init_reference_estimators()
        self.init_algorithm_parameter_inputs()
        self.sl.insertLayout(0, self.al)

        self.hl.insertLayout(0, self.sl)

        self.error_label = QtGui.QLabel('Mean Error:')
        self.vl.insertWidget(-1, self.error_label)

        self.save_button = QtGui.QPushButton('Save Result')
        self.vl.addWidget(self.save_button)

        self.render_actors(self.registered_actors)
        self.vtkWidget.GetRenderWindow().Render()
        self.ren.ResetCamera()

        self.update_actor_visibility()
        self.calc_button.clicked.connect(self.calculate)
        self.save_button.clicked.connect(self.store_data)

    def get_actor_for_property(self, property, outline):
        points = outline[property]
        num_points = points.shape[0]
        edges = np.zeros((num_points, 2), dtype=np.int)
        edges[:, 0] = range(num_points)
        edges[:, 1] = range(1, num_points+1)
        edges[-1, 1] = 0
        return get_outline_actor({
            'points': points,
            'edges': edges
        }, outline['color'], 0xFFFF, False)

    def initialize_actors(self):
        self.registered_actors = map(partial(self.get_actor_for_property, 'points'), self.bones)

    def init_outline_checkboxes(self):
        layout = QtGui.QVBoxLayout()
        scroll_area = QtGui.QScrollArea()
        self.outlines_box = QtGui.QGroupBox('Selected Outlines')
        self.outlines_buttons = map(lambda e: QtGui.QCheckBox(e['filename'].decode('utf-8')), self.bones)
        for b in self.outlines_buttons:
            b.toggled.connect(self.update_actor_visibility)
            layout.addWidget(b)
        self.outlines_box.setLayout(layout)
        scroll_area.setWidget(self.outlines_box)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(300)
        self.sl.addWidget(scroll_area)

    def init_class_selector(self):
        classes = list(set([b['class'] for b in self.bones]))
        class_labels = list(set([b['class_label'] for b in self.bones]))

        layout = QtGui.QVBoxLayout()
        self.classes_box = QtGui.QGroupBox('Select / Deselect Classes')
        for i, c in enumerate(classes):
            b = QtGui.QCheckBox(class_labels[i])

            def select_class(b, cls):
                try:
                    checked = b.isChecked()
                    to_check = map(lambda o: o['class'] == cls, self.bones)
                    for i in range(len(self.outlines_buttons)):
                        if to_check[i]:
                            self.outlines_buttons[i].setChecked(checked)
                except:
                    print(traceback.format_exc())

            b.toggled.connect(partial(select_class, b, c))
            b.toggle()
            layout.addWidget(b)
        self.classes_box.setLayout(layout)
        self.sl.insertWidget(0, self.classes_box)

    def init_estimators(self):
        layout = QtGui.QVBoxLayout()
        self.estimators_box = QtGui.QGroupBox('Transformation Type')
        self.estimators_buttons = map(lambda e: QtGui.QRadioButton(e['label']), self.estimators)
        self.estimators_buttons[0].setChecked(True)
        for b in self.estimators_buttons:
            layout.addWidget(b)
        self.estimators_box.setLayout(layout)
        self.al.addWidget(self.estimators_box)

    def init_reference_estimators(self):
        layout = QtGui.QVBoxLayout()
        self.reference_estimators_box = QtGui.QGroupBox('Selected Reference Points')
        self.reference_estimators_buttons = map(lambda e: QtGui.QRadioButton(e['label']), self.reference_estimators)
        self.reference_estimators_buttons[0].setChecked(True)
        for b in self.reference_estimators_buttons:
            layout.addWidget(b)
        self.reference_estimators_box.setLayout(layout)
        self.al.addWidget(self.reference_estimators_box)

    def init_algorithm_parameter_inputs(self):
        layout = QtGui.QFormLayout()
        self.parameters_box = QtGui.QGroupBox('Algorithm Parameters')
        self.iterations_button = QtGui.QSpinBox()
        self.iterations_button.setValue(1)
        layout.addRow('Iterations', self.iterations_button)

        self.independent_scaling_button = QtGui.QCheckBox()
        layout.addRow('Axis independent scaling', self.independent_scaling_button)

        self.continue_registration_button = QtGui.QCheckBox()
        layout.addRow('Continue begun registration', self.continue_registration_button)

        self.parameters_box.setLayout(layout)
        self.al.addWidget(self.parameters_box)

    @error_decorator
    def calculate(self):
        self.calc_button.setEnabled(False)
        QtGui.QApplication.processEvents()

        bones = list([ o for i, o in enumerate(self.bones) if self.outlines_buttons[i].isChecked() ])
        estimator = list([ e for i, e in enumerate(self.estimators) if self.estimators_buttons[i].isChecked() ])[0]['fn']
        reference_estimator = list([ r for i, r in enumerate(self.reference_estimators) if self.reference_estimators_buttons[i].isChecked() ])[0]['fn']
        iterations = self.iterations_button.value()
        independent_scaling = self.independent_scaling_button.isChecked()
        continue_registration = self.continue_registration_button.isChecked()

        self.register_fn(bones, estimator, reference_estimator, iterations, progress_callback=self.update_progress_bar, independent_scaling=independent_scaling, continue_registration=continue_registration)

        QtGui.QApplication.processEvents()
        self.calc_button.setEnabled(True)

        self.update_actor_data()
        self.update_info_panel()
        self.initial_registration_done = True

    def update_progress_bar(self, progress, max_progress):
        self.progress_bar.setMaximum(max_progress)
        self.progress_bar.setValue(progress)
        QtGui.QApplication.processEvents()

    def update_actor_visibility(self):
        for i, actor in enumerate(self.registered_actors):
            if self.outlines_buttons[i].isChecked():
                actor.VisibilityOn()
            else:
                actor.VisibilityOff()
        self.vtkWidget.GetRenderWindow().Render()

    def update_actor_data(self):
        for actor in self.registered_actors:
            self.ren.RemoveActor(actor)
        self.registered_actors = map(partial(self.get_actor_for_property, 'registered'), self.bones)
        for actor in self.registered_actors:
            self.ren.AddActor(actor)
        self.update_actor_visibility()
        if not self.initial_registration_done:
            self.ren.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def update_info_panel(self):
        mean_error = np.mean([ o['error'] for o in self.bones ])
        self.error_label.setText('Mean Error: {0}'.format(mean_error))

    def get_outlines_in_save_format(self):
        def get_save_format(outline):
            num_points = outline['registered'].shape[0]
            edges = np.zeros((num_points, 2))
            edges[:, 0] = range(0, num_points)
            edges[:, -1] = range(1, num_points+1)
            edges[-1, 1] = 0
            # Fixme: Handle markers and missing markers
            #markers = dict([(i,outline['registered_markers'][i-1,:]) for i in range(1, 12)])

            points = outline['registered']
            # Fixme: Do Normalization for Direction only
            # points, edges = gh.normalize_outline(outline['registered'], edges)

            outline_to_save = {
                'filename': outline['filename'],
                'done': True,
                'points': points,
                #    'markers': markers,
                'edges': edges
            }

            return outline_to_save

        return list(map(get_save_format, self.bones))

    def store_data(self):
        directory = str(QtGui.QFileDialog.getExistingDirectory(
            self,
            'Open Directory for Comparison',
            os.getcwd(),
            QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks
        ))

        if len(directory) > 0:
            lh.save_files(directory, self.get_outlines_in_save_format())
