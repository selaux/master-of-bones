import vtk
from PyQt4 import QtGui
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


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
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
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
        self.ren.AddActor(scale)

        self.ren.ResetCamera()

        self.showMaximized()
        self.show()
        self.activateWindow()
        self.iren.Initialize()