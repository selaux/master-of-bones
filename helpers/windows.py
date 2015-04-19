import vtk
from functools import partial
from PyQt4 import QtCore, QtGui
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class VTKWindow(QtGui.QMainWindow):

    def __init__(self, parent = None, title=None):
        QtGui.QMainWindow.__init__(self, parent)
        if title:
            self.setWindowTitle(title)
        else:
            self.setWindowTitle('Visualization')

        self.vl = QtGui.QVBoxLayout()
        self.frame = QtGui.QFrame()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

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

    def render_actors(self, actors, legends=[]):
        for actor in actors:
            self.ren.AddActor(actor)

        if len(legends) > 0:
            legend = vtk.vtkLegendBoxActor()
            legend.SetNumberOfEntries(len(legends))
            legend.GetEntryTextProperty().SetFontSize(25)
            legend.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            legend.GetPositionCoordinate().SetValue(0, 0);
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
        scale.GetRightAxis().GetProperty().SetColor(0, 0, 0);
        scale.GetRightAxis().GetLabelTextProperty().SetFontSize(10)
        scale.GetRightAxis().GetLabelTextProperty().ShadowOff()
        scale.GetRightAxis().GetLabelTextProperty().SetColor(0.2, 0.2, 0.2)
        scale.GetTopAxis().GetProperty().SetColor(0, 0, 0);
        scale.GetTopAxis().GetLabelTextProperty().SetFontSize(10)
        scale.GetTopAxis().GetLabelTextProperty().ShadowOff()
        scale.GetTopAxis().GetLabelTextProperty().SetColor(0.2, 0.2, 0.2)
        self.ren.AddActor(scale)

        self.ren.ResetCamera()

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.showMaximized()
        self.show()
        self.activateWindow()
        self.iren.Initialize()

def create_single_legend_actor(color, line_style):
    points = vtk.vtkPoints()
    raw_points = [
         [0,0],
         [1,0],
         [0,0.25],
         [1,0.25],
         [0, 0.5],
         [1, 0.5],
         [0, 0.75],
         [1, 0.75],
         [0,1],
         [1,1],
         [0.5, 1],
         [0.5, 0]
    ]
    for point in raw_points:
        points.InsertNextPoint([ point[1], point[0], 1.0 ])

    p1 = [
        [0,2,3],
        [0,3,1]
    ]
    p2 = [
        [2,4,5],
        [2,5,3]
    ]
    p3 = [
        [4,6,7],
        [4,7,5]
    ]
    p4 = [
        [6,8,9],
        [6,9,7]
    ]
    triangles_raw = [
        [0, 10, 11],
        [0, 9, 10]
    ]
    triangles_raw += p1 + p2 + p3 + p4
    if line_style == 0xF0F0:
        triangles_raw = p1 + p3
    elif line_style == 0xFF00:
        triangles_raw = p1 + p2
    elif line_style == 0x000F:
        triangles_raw = p4
    triangles = vtk.vtkCellArray()
    for t in triangles_raw:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, t[0])
        triangle.GetPointIds().SetId(1, t[1])
        triangle.GetPointIds().SetId(2, t[2])
        triangles.InsertNextCell(triangle)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)
    return polydata

def show_window(window):
    app = QtCore.QCoreApplication.instance()
    if not hasattr(app, 'references') or not isinstance(app.references, set):
        app.references = set()
    app.references.add(window)
    def remove(app, window, event):
        app.references.remove(window)
    window.connect(window, QtCore.SIGNAL('triggered()'), partial(remove, app, window))
