from functools import partial
import os
from inspect import getargspec
import traceback
import vtk
from PyQt4 import QtCore, QtGui
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def error_decorator(fn):
    inspected = getargspec(fn)
    num_args = len(inspected.args) - 1

    def catched_fn(self, *args):
        try:
            fn(self, *args[:num_args])
        except SystemExit:
            self.close()
        except:
            QtGui.QMessageBox.critical(
                self,
                'An Error Occured',
                '<b>An unexpected error occured, please report it to the programmer</b><br><code>{0}</code>'.format(traceback.format_exc())
            )
    return catched_fn


class VTKWindow(QtGui.QMainWindow):
    def __init__(self, parent = None, title=None):
        QtGui.QMainWindow.__init__(self, parent)
        if title:
            self.setWindowTitle(title)
        else:
            self.setWindowTitle('Visualization')

        self.vl = QtGui.QVBoxLayout()
        self.hl = QtGui.QHBoxLayout()
        self.frame = QtGui.QFrame()
        self.vtkWidget, self.ren, self.iren, self.intstyle = self.init_vtk_widget(self.frame)
        self.save_main_vizualization_button = self.init_save_vizualization_button(
            self.frame,
            self.vtkWidget.GetRenderWindow()
        )
        self.vl.addWidget(self.vtkWidget)
        self.hl.insertLayout(-1, self.vl)

        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.vtkWidget.GetRenderWindow().LineSmoothingOn()
        self.vtkWidget.GetRenderWindow().PolygonSmoothingOn()
        self.vtkWidget.GetRenderWindow().PointSmoothingOn()
        self.vtkWidget.GetRenderWindow().SetMultiSamples(8)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.ren.GetActiveCamera().ParallelProjectionOn()
        self.intstyle = vtk.vtkInteractorStyleRubberBand2D()
        self.iren.SetInteractorStyle(self.intstyle)
        self.ren.SetBackground(1.0, 1.0, 1.0)

        self.frame.setLayout(self.hl)
        self.setCentralWidget(self.frame)

    def init_vtk_widget(self, widget):
        vtk_widget = QVTKRenderWindowInteractor(widget)
        renderer = vtk.vtkRenderer()
        vtk_widget.GetRenderWindow().AddRenderer(renderer)

        renderer.SetBackground(1.0, 1.0, 1.0)

        interactor = vtk_widget.GetRenderWindow().GetInteractor()
        interactor_style = vtk.vtkInteractorStyleRubberBand2D()
        interactor.SetInteractorStyle(interactor_style)

        vtk_widget.GetRenderWindow().LineSmoothingOn()
        vtk_widget.GetRenderWindow().PolygonSmoothingOn()
        vtk_widget.GetRenderWindow().PointSmoothingOn()
        vtk_widget.GetRenderWindow().SetMultiSamples(8)

        return vtk_widget, renderer, interactor, interactor_style

    def init_scale(self):
        scale = vtk.vtkLegendScaleActor()
        scale.SetLabelModeToXYCoordinates()
        scale.LegendVisibilityOff()
        scale.LeftAxisVisibilityOff()
        scale.BottomAxisVisibilityOff()
        scale.SetRightBorderOffset(50)
        scale.GetRightAxis().GetProperty().SetColor(0, 0, 0)
        scale.GetRightAxis().GetLabelTextProperty().SetFontSize(10)
        scale.GetRightAxis().GetLabelTextProperty().ShadowOff()
        scale.GetRightAxis().GetLabelTextProperty().SetColor(0.2, 0.2, 0.2)
        scale.GetTopAxis().GetProperty().SetColor(0, 0, 0)
        scale.GetTopAxis().GetLabelTextProperty().SetFontSize(10)
        scale.GetTopAxis().GetLabelTextProperty().ShadowOff()
        scale.GetTopAxis().GetLabelTextProperty().SetColor(0.2, 0.2, 0.2)

        return scale

    @error_decorator
    def init_save_vizualization_button(self, widget, render_window):
        button = QtGui.QPushButton(widget)

        old_resize_event = widget.resizeEvent
        def new_resize_event(ev):
            button.move(widget.mapToGlobal(QtCore.QPoint(20, 20)))
            old_resize_event(ev)
        widget.resizeEvent = new_resize_event

        button.setGeometry(20, 20, 32, 32)
        button.setIcon(QtGui.QIcon.fromTheme('document-save'))
        button.setToolTip('Save Visualization')

        button.clicked.connect(partial(self.store_visualization, render_window))

        return button

    @error_decorator
    def store_visualization(self, render_window):
        file_name = str(QtGui.QFileDialog.getSaveFileName(
            self,
            'Save Visualization',
            os.path.join(os.getcwd(), '/visualization.svg'),
            'SVG-Images (*.svg)'
        ))

        if len(file_name) > 0:
            exporter = vtk.vtkGL2PSExporter()
            exporter.SetRenderWindow(render_window)
            exporter.SetFileFormatToSVG()
            exporter.CompressOff()
            exporter.DrawBackgroundOff()
            exporter.SetFilePrefix(os.path.splitext(file_name)[0])
            exporter.Write()

    def render_actors(self, actors, legends=[]):
        for actor in actors:
            self.ren.AddActor(actor)

        if len(legends) > 0:
            legend = vtk.vtkLegendBoxActor()
            legend.SetNumberOfEntries(len(legends))
            legend.GetEntryTextProperty().SetFontSize(25)
            legend.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            legend.GetPositionCoordinate().SetValue(0, 0)
            legend.GetPosition2Coordinate().SetCoordinateSystemToDisplay()
            legend.GetPosition2Coordinate().SetValue(250, len(legends)*30)
            for i, l in enumerate(legends):
                legend.SetEntry(i, l[2], l[0], l[1])
            self.ren.AddActor(legend)

        self.ren.AddActor(self.init_scale())

        self.ren.ResetCamera()

        self.showMaximized()
        self.show()
        self.activateWindow()
        self.iren.Initialize()
