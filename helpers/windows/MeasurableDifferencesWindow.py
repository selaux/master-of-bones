from math import floor
from helpers.to_vtk import get_outline_actor, get_line_actor
from sklearn.externals.six import StringIO
import numpy as np
import tempfile
from PyQt4 import QtGui
import helpers.classes as ch
from VTKWindow import VTKWindow, error_decorator
from widgets import BoneList
import pydot
from sklearn import tree


class MeasurableDifferencesWindow(VTKWindow):
    def __init__(self, bones, compare_fn, landmark_extractors):
        VTKWindow.__init__(self, title='Find Measurable Differences Helper')

        self.bones = bones
        self.compare_fn = compare_fn
        self.landmark_extractors = landmark_extractors
        self.classes_in_data = list(set(map(lambda b: b['class'], self.bones)))

        self.al = QtGui.QVBoxLayout()

        self.init_compared_classes()
        self.init_landmarks_extractions()
        self.init_algorithm_params()
        self.init_visualization_selection()
        self.ratio_label = QtGui.QLabel('')
        self.vl.addWidget(self.ratio_label)

        self.calc_button = QtGui.QPushButton('Calculate')
        self.al.addWidget(self.calc_button)
        self.progress_bar = QtGui.QProgressBar()
        self.al.addWidget(self.progress_bar)

        self.init_result_display()
        self.graph_viz_label = QtGui.QLabel('')
        self.al.addWidget(self.graph_viz_label)

        self.actors = []
        self.render_actors(self.actors)

        self.hl.addLayout(self.al)

        self.calc_button.clicked.connect(self.calculate)
        self.bone_list.selectionModel().currentChanged.connect(self.update_visualization)
        self.landmark_ratio_list.selectionModel().currentChanged.connect(self.update_visualization)

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
        self.al.addWidget(self.classes_box)

    def init_landmarks_extractions(self):
        layout = QtGui.QVBoxLayout()
        self.landmarks_extraction_box = QtGui.QGroupBox('Window Extraction Type')
        self.landmarks_extraction_buttons = map(lambda e: QtGui.QRadioButton(e['label']), self.landmark_extractors)
        self.landmarks_extraction_buttons[0].setChecked(True)
        for b in self.landmarks_extraction_buttons:
            layout.addWidget(b)
        self.landmarks_extraction_box.setLayout(layout)
        self.al.addWidget(self.landmarks_extraction_box)

    def init_algorithm_params(self):
        layout = QtGui.QFormLayout()

        self.tree_criteria = [
            'entropy',
            'gini'
        ]
        self.tree_criterion_box = QtGui.QComboBox()
        for c in self.tree_criteria:
            self.tree_criterion_box.addItem(c)
        self.tree_criterion_box.setCurrentIndex(0)
        layout.addRow('Split Criterion', self.tree_criterion_box)

        self.max_depth_spinbox = QtGui.QSpinBox()
        self.max_depth_spinbox.setValue(3)
        layout.addRow('Max Decision Tree Depth', self.max_depth_spinbox)

        self.tree_min_samples_leaf_spinbox = QtGui.QSpinBox()
        self.tree_min_samples_leaf_spinbox.setValue(2)
        layout.addRow('Min Samples in Decision Tree Leaf', self.tree_min_samples_leaf_spinbox)

        self.num_folds_spinbox = QtGui.QSpinBox()
        self.num_folds_spinbox.setValue(floor(len(self.bones) / 2))
        layout.addRow('Number of Folds for Validation', self.num_folds_spinbox)

        self.al.addLayout(layout)

    def init_result_display(self):
        layout = QtGui.QFormLayout()

        self.mean_cv_score_label = QtGui.QLabel('')
        layout.addRow('Mean Cross-Validation-Score', self.mean_cv_score_label)

        self.cv_confidence_interval_label = QtGui.QLabel('')
        # layout.addRow('95% confidence interval', self.cv_confidence_interval_label)

        self.al.addLayout(layout)

    def init_visualization_selection(self):
        layout = QtGui.QHBoxLayout()

        self.bone_list = BoneList(self.bones, self)
        layout.addWidget(self.bone_list)

        self.landmark_ratio_list = QtGui.QListWidget()
        layout.addWidget(self.landmark_ratio_list)

        self.vl.insertLayout(0, layout)

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

    @error_decorator
    def calculate(self):
        tree_criterion = self.tree_criteria[self.tree_criterion_box.currentIndex()]
        tree_max_depth = self.max_depth_spinbox.value()
        tree_min_samples_leaf = self.tree_min_samples_leaf_spinbox.value()
        n_folds = self.num_folds_spinbox.value()
        class1, class2 = self.get_compared_classes()
        landmark_extractor = list([f for i, f in enumerate(self.landmark_extractors) if self.landmarks_extraction_buttons[i].isChecked()])[0]['fn']

        kwargs = {
            'tree_criterion': tree_criterion,
            'tree_max_depth': tree_max_depth,
            'tree_min_samples_leaf': tree_min_samples_leaf,
            'landmark_extractor': landmark_extractor,
            'n_folds': n_folds,
            'progress_callback': self.update_progress_bar
        }

        self.calc_button.setEnabled(False)
        self.result = self.compare_fn(self.bones, class1, class2, **kwargs)

        self.update_tree_viz()
        self.update_landmark_ratio_list()
        self.update_visualization()

        self.calc_button.setEnabled(True)

    def update_progress_bar(self, progress, max_progress):
        self.progress_bar.setMaximum(max_progress)
        self.progress_bar.setValue(progress)
        QtGui.QApplication.processEvents()

    def update_tree_viz(self):
        if self.result:
            dot_data = StringIO()
            tree.export_graphviz(self.result['decision_tree'], out_file=dot_data)
            graph = pydot.graph_from_dot_data(dot_data.getvalue())

            with tempfile.NamedTemporaryFile(suffix=".png") as fout:
                graph.write(fout.name, format="png")
                pixmap = QtGui.QPixmap(fout.name)
                self.graph_viz_label.setPixmap(pixmap)

            self.mean_cv_score_label.setText(str(self.result['mean_cv_score']))
            self.cv_confidence_interval_label.setText(str(self.result['cv_confidence_interval']))

    def update_landmark_ratio_list(self):
        self.landmark_ratio_list.clear()
        if self.result:
            combinations = self.result['landmark_combinations']
            landmark_labels = ["{}: {}".format(str(i), str(r)) for i, r in enumerate(combinations)]
            for label in landmark_labels:
                self.landmark_ratio_list.addItem(label)
            self.landmark_ratio_list.setCurrentRow(0)

    @error_decorator
    def update_visualization(self):
        current_bone = self.bones[self.bone_list.currentIndex().row()]
        landmark_combinations = self.result['landmark_combinations'][self.landmark_ratio_list.currentRow()]
        landmark_combination_1 = landmark_combinations[0]
        landmark_combination_2 = landmark_combinations[1]
        ratio = current_bone['landmark_distances'][landmark_combination_1] / current_bone['landmark_distances'][landmark_combination_2]

        self.ratio_label.setText("Landmark-Distances Ratio: {}".format(ratio))

        for actor in self.actors:
            self.ren.RemoveActor(actor)
        self.actors = []

        outline_actor = get_outline_actor({
            'points': current_bone['points'],
            'edges': current_bone['edges'].astype(np.int)
        }, (0, 0, 0), 0xFFFF, False)
        landmark_combination_1_actor = get_line_actor(np.array([
            current_bone['landmarks'][landmark_combination_1[0]],
            current_bone['landmarks'][landmark_combination_1[1]]
        ]))
        landmark_combination_1_actor.GetProperty().SetRepresentationToWireframe()
        landmark_combination_1_actor.GetProperty().SetLineWidth(2)
        landmark_combination_1_actor.GetProperty().SetColor(255, 0, 0)
        landmark_combination_2_actor = get_line_actor(np.array([
            current_bone['landmarks'][landmark_combination_2[0]],
            current_bone['landmarks'][landmark_combination_2[1]]
        ]))
        landmark_combination_2_actor.GetProperty().SetRepresentationToWireframe()
        landmark_combination_2_actor.GetProperty().SetLineWidth(2)
        landmark_combination_2_actor.GetProperty().SetColor(0, 255, 0)

        self.actors.append(outline_actor)
        self.actors.append(landmark_combination_1_actor)
        self.actors.append(landmark_combination_2_actor)
        for actor in self.actors:
            self.ren.AddActor(actor)

        self.ren.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
