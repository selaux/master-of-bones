import vtk
import numpy as np
import gzip
import pickle
import traceback
from PyQt4 import QtCore, QtGui
from ..to_vtk import get_outline_actor
from .. import geometry as gh
from VTKWindow import VTKWindow


class TriangulationWindow(VTKWindow):
    def __init__(self, bones, do_triangulation):
        VTKWindow.__init__(self, title='Triangulation Helper')

        self.bones = bones
        self.do_triangulation = do_triangulation
        self.current = bones[0]
        self.last_step = None
        self.current_modified = False

        self.list_model = lm = BonesListModel(self.bones, self)
        self.list_view = QtGui.QListView()
        self.list_view.setModel(lm)
        self.list_view.selectionModel().select(lm.index(0), QtGui.QItemSelectionModel.SelectCurrent)
        self.list_view.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.MinimumExpanding)
        self.hl.insertWidget(0, self.list_view)

        self.bl = QtGui.QHBoxLayout()
        self.toggle_triangulation_button = self.get_new_button('Toggle Triangulation', 'image-x-generic', checkable=True)
        self.bl.insertWidget(-1, self.toggle_triangulation_button)
        self.toggle_triangulation_button.setChecked(True)
        self.toggle_markers_button = self.get_new_button('Toggle Markers', 'text-x-generic', checkable=True)
        self.bl.insertWidget(-1, self.toggle_markers_button)
        self.toggle_area_effect_button = self.get_new_button('Toggle Area Effect', 'list-add', checkable=True)
        self.bl.insertWidget(-1, self.toggle_area_effect_button)
        self.undo_button = self.get_new_button('Undo Last Step', 'edit-undo')
        self.bl.insertWidget(-1, self.undo_button)
        self.show_outline_button = self.get_new_button('Show Outline', 'zoom-in', checkable=True)
        self.bl.insertWidget(-1, self.show_outline_button)
        self.mark_as_done_button = self.get_new_button('Mark as done', 'emblem-readonly', checkable=True)
        self.bl.insertWidget(-1, self.mark_as_done_button)
        self.mark_as_done_button.setChecked('done' in self.current and self.current['done'])
        self.save_button = self.get_new_button('Save triangulation', 'document-save')
        self.bl.insertWidget(-1, self.save_button)
        self.vl.insertLayout(-1, self.bl)

        self.image_import = vtk.vtkImageImport()
        self.image_actor = vtk.vtkImageActor()
        self.triangle_data = vtk.vtkPolyData()
        self.triangle_mapper = vtk.vtkPolyDataMapper()
        self.triangle_actor = vtk.vtkActor()
        self.marker_data = vtk.vtkPolyData()
        self.marker_mapper = vtk.vtkPolyDataMapper()
        self.marker_actor = vtk.vtkActor()
        self.marker_actor.VisibilityOff()
        self.marker_labels_filter = vtk.vtkPointSetToLabelHierarchy()
        self.marker_labels_mapper = vtk.vtkLabelPlacementMapper()
        self.marker_labels_actor = vtk.vtkActor2D()
        self.marker_labels_actor.VisibilityOff()
        self.initialize_actors()
        self.render_actors([
            self.image_actor,
            self.triangle_actor,
            self.marker_actor,
            self.marker_labels_actor
        ])
        self.update_current_data(including_image=True)
        self.vtkWidget.GetRenderWindow().Render()
        self.ren.ResetCamera()

        self.iren.RemoveObservers('CharEvent')
        self.iren.AddObserver('KeyPressEvent', self.on_key, 0.0)
        self.intstyle.AddObserver('SelectionChangedEvent', self.area_selected, 0.0)
        self.list_view.selectionModel().currentChanged.connect(self.on_bone_model_change)
        self.toggle_triangulation_button.clicked.connect(self.toggle_triangulation)
        self.toggle_markers_button.clicked.connect(self.toggle_markers)
        self.mark_as_done_button.clicked.connect(self.mark_as_done)
        self.save_button.clicked.connect(self.save_current)
        self.undo_button.clicked.connect(self.undo)
        self.show_outline_button.clicked.connect(self.trigger_outline)

    def get_new_button(self, label, icon, checkable=False):
        button = QtGui.QPushButton('')
        button.setToolTip(label)
        button.setIcon(QtGui.QIcon.fromTheme(icon))
        button.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        if checkable:
            button.setCheckable(True)
        return button

    def initialize_actors(self):
        self.image_actor.SetInputData(self.image_import.GetOutput())

        self.triangle_mapper.SetInputData(self.triangle_data)
        self.triangle_actor.SetMapper(self.triangle_mapper)
        self.triangle_actor.GetProperty().SetRepresentationToWireframe()
        self.triangle_actor.GetProperty().SetLineWidth(1.5)

        self.outline_actor = None

        self.marker_mapper.SetInputData(self.marker_data)
        self.marker_actor.SetMapper(self.marker_mapper)
        self.marker_actor.GetProperty().SetPointSize(10)
        self.marker_actor.GetProperty().SetColor(1.0, 0, 0)
        self.marker_labels_filter.SetInputDataObject(self.marker_data)
        self.marker_labels_filter.SetLabelArrayName("labels")
        self.marker_labels_filter.GetTextProperty().SetFontSize(16)
        self.marker_labels_filter.GetTextProperty().BoldOn()
        self.marker_labels_filter.GetTextProperty().SetColor(1.0, 0, 0)
        self.marker_labels_filter.GetTextProperty().SetLineOffset(20)
        self.marker_labels_mapper.SetInputConnection(self.marker_labels_filter.GetOutputPort())
        self.marker_labels_actor.SetMapper(self.marker_labels_mapper)


    def update_current_data(self, including_image):
        tri = self.do_triangulation(self.current['bone_pixels'])

        outline_points, outline_edges = gh.extract_outline(tri.points, tri.simplices)
        if self.outline_actor:
            self.ren.RemoveActor(self.outline_actor)
        self.outline_actor = get_outline_actor({
            'points': outline_points,
            'edges': outline_edges
        }, (255, 0, 0), 0xFFFF, False)
        self.ren.AddActor(self.outline_actor)
        if self.show_outline_button.isChecked():
            self.outline_actor.VisibilityOn()
        else:
            self.outline_actor.VisibilityOff()

        points = vtk.vtkPoints()
        for point in tri.points:
            points.InsertNextPoint([ point[1], point[0], 1.0 ])
        triangles = vtk.vtkCellArray()
        for sim in tri.simplices:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, sim[0])
            triangle.GetPointIds().SetId(1, sim[1])
            triangle.GetPointIds().SetId(2, sim[2])
            triangles.InsertNextCell(triangle)
        self.triangle_data.SetPoints(points)
        self.triangle_data.SetPolys(triangles)
        self.triangle_data.Modified()

        marker_points = vtk.vtkPoints()
        marker_vertices = vtk.vtkCellArray()
        labels = vtk.vtkStringArray()
        labels.SetNumberOfValues(len(self.current['markers'].values()))
        labels.SetName("labels")
        for i, marker_no in enumerate(self.current['markers']):
            marker = self.current['markers'][marker_no]
            marker_points.InsertNextPoint([ marker[1], marker[0], 1.0 ])
            labels.SetValue(i, str(marker_no))
            marker_vertices.InsertNextCell(1)
            marker_vertices.InsertCellPoint(i)
        self.marker_data.SetPoints(marker_points)
        self.marker_data.SetVerts(marker_vertices)
        self.marker_data.GetPointData().AddArray(labels)
        self.marker_data.Modified()
        self.marker_labels_filter.Update()

        if including_image:
            image = np.flipud(self.current['image'].copy()).astype('uint8')
            data_string = image.tostring()
            self.image_import.CopyImportVoidPointer(data_string, len(data_string))
            self.image_import.SetDataScalarTypeToUnsignedChar()
            self.image_import.SetNumberOfScalarComponents(3)
            self.image_import.SetDataExtent(0,image.shape[1]-1,
                                        0,image.shape[0]-1,
                                        0,0)
            self.image_import.SetWholeExtent(0,image.shape[1]-1,
                                         0,image.shape[0]-1,
                                         0,0)
            self.image_import.Update()
            self.image_import.Modified()

    def closeEvent(self, event):
        self.ask_for_save()
        VTKWindow.closeEvent(self, event)

    def toggle_triangulation(self, state):
        try:
            if state:
                self.triangle_actor.VisibilityOn()
            else:
                self.triangle_actor.VisibilityOff()
            self.vtkWidget.GetRenderWindow().Render()
        except:
            print(traceback.format_exc())

    def toggle_markers(self, state):
        try:
            if state:
                self.marker_actor.VisibilityOn()
                self.marker_labels_actor.VisibilityOn()
            else:
                self.marker_actor.VisibilityOff()
                self.marker_labels_actor.VisibilityOff()
            self.vtkWidget.GetRenderWindow().Render()
        except:
            print(traceback.format_exc())

    def mark_as_done(self, state):
        old = 'done' in self.current and self.current['done']
        try:
            index = self.bones.index(self.current)
            self.current['done'] = state
            if old != state:
                self.current_modified = True
                self.list_model.dataChanged.emit(self.list_model.index(index), self.list_model.index(index))
        except:
            print(traceback.format_exc())

    def trigger_outline(self):
        if self.outline_actor:
            if self.show_outline_button.isChecked():
                self.outline_actor.VisibilityOn()
            else:
                self.outline_actor.VisibilityOff()
            self.vtkWidget.GetRenderWindow().Render()

    def save_current(self):
        try:
            filename = self.current['save_path']
            tri = self.do_triangulation(self.current['bone_pixels'])
            to_save = {
                'done': 'done' in self.current and self.current['done'],
                'bone_pixels': self.current['bone_pixels'],
                'markers': self.current['markers'],
                'points': tri.points,
                'simplices': tri.simplices
            }
            with gzip.open(filename, 'wb') as f:
                pickle.dump(to_save, f)
            self.current_modified = False
        except:
            print(traceback.format_exc())

    def undo(self):
        try:
            if self.last_step is not None:
                self.current['bone_pixels'] = self.last_step
                self.last_step = None

                self.current_modified = True
                self.update_current_data(including_image=False)
                self.vtkWidget.GetRenderWindow().Render()
        except:
            print(traceback.format_exc())

    def on_bone_model_change(self, current, last):
        try:
            old = self.current
            new = self.bones[current.row()]

            self.ask_for_save()
            self.current = new
            self.last_step = None
            self.current_modified = False
            self.update_current_data(including_image=True)
            self.ren.ResetCamera()
            self.vtkWidget.GetRenderWindow().Render()

            self.toggle_triangulation_button.setChecked(True)
            self.toggle_triangulation(True)
            self.toggle_markers_button.setChecked(False)
            self.toggle_markers(False)
            self.mark_as_done_button.setChecked('done' in self.current and self.current['done'])
        except:
            print(traceback.format_exc())

    def ask_for_save(self, *args):
        if self.current_modified:
            reply = QtGui.QMessageBox.question(self,
                                               'Do you want to save?',
                                               'The current mesh has been modified. Do you want to save it?',
                                               QtGui.QMessageBox.Yes,
                                               QtGui.QMessageBox.No)
            if reply == QtGui.QMessageBox.Yes:
                self.save_current()

    def reshape_bone_pixels_if_necessary(self, x, y):
        bone_pixels = self.current['bone_pixels']

        print(y, bone_pixels.shape[0])
        if y <= 0:
            print("IncSize", bone_pixels.shape[0], bone_pixels.shape[0]-y)
            old = bone_pixels
            bone_pixels = np.zeros((bone_pixels.shape[0]-y, old.shape[1]))
            bone_pixels[-y:old.shape[0]-y, :] = old
        if x >= bone_pixels.shape[1]:
            print("IncSize", bone_pixels.shape[1], x)
            old = bone_pixels
            bone_pixels = np.zeros((old.shape[0], x+1))
            bone_pixels[:, 0:old.shape[1]] = old

        self.current['bone_pixels'] = bone_pixels

        return bone_pixels

    def area_selected(self, *args):
        try:
            start_display_x, start_display_y = self.intstyle.GetStartPosition()
            end_display_x, end_display_y = self.intstyle.GetEndPosition()

            self.ren.SetDisplayPoint((start_display_x, start_display_y, 0))
            self.ren.DisplayToWorld()
            start_world_x, start_world_y, z, w = self.ren.GetWorldPoint()

            self.ren.SetDisplayPoint((end_display_x, end_display_y, 0))
            self.ren.DisplayToWorld()
            end_world_x, end_world_y, z, w = self.ren.GetWorldPoint()

            low_x, low_y = min(start_world_x, end_world_x), min(start_world_y, end_world_y)
            high_x, high_y = max(start_world_x, end_world_x), max(start_world_y, end_world_y)



            self.reshape_bone_pixels_if_necessary(low_x, low_y)
            bone_pixels = self.reshape_bone_pixels_if_necessary(high_x, high_y)

            low_y, high_y = bone_pixels.shape[0] - high_y, bone_pixels.shape[0] - low_y

            self.last_step = bone_pixels.copy()
            bone_pixels[low_y:high_y, low_x:high_x] = 0
            if self.toggle_area_effect_button.isChecked():
                bone_pixels[low_y:high_y:25, low_x:high_x:25] = 255

            self.current_modified = True
            self.update_current_data(including_image=False)
            self.vtkWidget.GetRenderWindow().Render()
        except:
            print(traceback.format_exc())

    def on_key(self, obj, ev):
        try:
            key = obj.GetKeyCode()
            triangulation_keymap = {
                'y': 255,
                'x': 0
            }
            marker_keymap = {
                '1': 1,
                '2': 2,
                '3': 3,
                '4': 4,
                '5': 5,
                '6': 6,
                '7': 7,
                '8': 8,
                '9': 9,
                '0': 10,
                ',': 11
            }

            if key and key in triangulation_keymap:
                mouse_position = obj.GetEventPosition()
                #print(mouse_position)
                if mouse_position:
                    bone_pixels = self.current['bone_pixels']

                    self.ren.SetDisplayPoint((mouse_position[0], mouse_position[1], 0))
                    self.ren.DisplayToWorld()
                    x, y, z, w = self.ren.GetWorldPoint()

                    set_to = triangulation_keymap[key]
                    if set_to == 0:
                        # Delete closest non zero pixel
                        indices_of_bone_pixels = np.nonzero(bone_pixels)
                        indices_of_bone_pixels = np.vstack(indices_of_bone_pixels).transpose()
                        indices_of_bone_pixels[:, 0] = bone_pixels.shape[0] - indices_of_bone_pixels[:, 0]
                        index_of_click = np.tile(np.array([y, x]), (indices_of_bone_pixels.shape[0], 1))
                        distances = np.linalg.norm(indices_of_bone_pixels - index_of_click, axis=1)

                        index_to_set = distances.argmin()
                        y = bone_pixels.shape[0] - indices_of_bone_pixels[index_to_set][0]
                        x = indices_of_bone_pixels[index_to_set][1]
                    else:
                        y = bone_pixels.shape[0] - y

                    x = int(x)
                    y = int(y)
                    bone_pixels = self.reshape_bone_pixels_if_necessary(x, y)

                    self.last_step = bone_pixels.copy()
                    bone_pixels[y, x] = set_to
                    self.current['bone_pixels'] = bone_pixels

                    self.current_modified = True
                    self.update_current_data(including_image=False)
                    self.vtkWidget.GetRenderWindow().Render()

                    return False
            if key and key in marker_keymap:
                mouse_position = obj.GetEventPosition()
                marker_no = marker_keymap[key]
                if mouse_position:
                    self.ren.SetDisplayPoint((mouse_position[0], mouse_position[1], 0))
                    self.ren.DisplayToWorld()
                    x, y, z, w = self.ren.GetWorldPoint()

                    self.current['markers'][marker_no] = np.array([int(y), int(x)])

                    self.current_modified = True
                    self.update_current_data(including_image=False)
                    self.vtkWidget.GetRenderWindow().Render()

                    return False
        except:
            print(traceback.format_exc())


class BonesListModel(QtCore.QAbstractListModel):
    def __init__(self, datain, parent=None):
        QtCore.QAbstractListModel.__init__(self, parent)
        self.listdata = datain

    def rowCount(self, parent):
        return len(self.listdata)

    def data(self, index, role):
        if role == QtCore.Qt.DecorationRole:
            row = index.row()
            edited = self.listdata[row]['edited']
            done = 'done' in self.listdata[row] and self.listdata[row]['done']
            class_label = self.listdata[row]['class_short']

            color = QtGui.QColor(255, 0, 0)
            if done:
                color = QtGui.QColor(0, 255, 0)
            elif edited:
                color = QtGui.QColor(255, 140, 0)

            pixmap = QtGui.QPixmap(26, 26)
            painter = QtGui.QPainter()
            pixmap.fill(color)
            painter.begin(pixmap)
            painter.setFont(QtGui.QFont('Helvetica', 10))
            painter.drawText(QtCore.QRectF(0, 0, 26, 26), QtCore.Qt.AlignCenter, class_label.decode('utf-8'))
            painter.end()

            return QtGui.QIcon(pixmap)
        if role == QtCore.Qt.DisplayRole:
            row = index.row()
            return self.listdata[row]['filename'].decode('utf-8')