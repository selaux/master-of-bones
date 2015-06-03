import vtk
import numpy as np

def get_points_actor(pts):
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    for i, p in enumerate(pts):
        points.InsertNextPoint([p[1], p[0], 1])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetVerts(vertices)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(10)
    actor.GetProperty().SetColor(1.0, 0, 0)

    return actor

def get_line_actor(pts):
    points = vtk.vtkPoints()
    for p in pts:
        points.InsertNextPoint([p[1], p[0], 1])

    lines = vtk.vtkCellArray()
    for i in range(0, pts.shape[0]-1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i+1)
        lines.InsertNextCell(line)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

def get_outline_actor(outline, base_color, line_style, show_direction):
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")

    points = vtk.vtkPoints()
    for point in outline['points']:
        points.InsertNextPoint([ point[1], point[0], 1.0 ])
    lines = vtk.vtkCellArray()

    for i, edge in enumerate(outline['edges']):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, edge[0])
        line.GetPointIds().SetId(1, edge[1])
        lines.InsertNextCell(line)

        color = tuple([int(255 * c) for c in base_color])
        if show_direction:
            color = (
                color[0] - color[0] * i / len(outline['edges']),
                color[1] - color[1] * i / len(outline['edges']),
                color[2] - color[2] * i / len(outline['edges'])
            )
        colors.InsertNextTuple3(*color)

    linesPolyData = vtk.vtkPolyData()
    linesPolyData.SetPoints(points)
    linesPolyData.SetLines(lines)
    linesPolyData.GetCellData().SetScalars(colors)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(linesPolyData)

    actor = vtk.vtkActor()
    actor.GetProperty().SetLineWidth(2.0)
    actor.GetProperty().SetLineStipplePattern(line_style)
    actor.SetMapper(mapper)

    return actor


def to_vtk_image(image):
    image = np.flipud(image.copy()).astype('uint8')

    imageString = image.tostring()
    imageImporter = vtk.vtkImageImport()
    imageImporter.CopyImportVoidPointer(imageString, len(imageString))
    imageImporter.SetDataScalarTypeToUnsignedChar()
    imageImporter.SetNumberOfScalarComponents(3)
    imageImporter.SetDataExtent(0, image.shape[1]-1,
                                0, image.shape[0]-1,
                                0, 0)
    imageImporter.SetWholeExtent(0, image.shape[1]-1,
                                 0, image.shape[0]-1,
                                 0, 0)
    imageImporter.Update()

    return imageImporter


def create_single_legend_actor(color, line_style):
    points = vtk.vtkPoints()
    raw_points = [
         [0, 0],
         [1, 0],
         [0, 0.25],
         [1, 0.25],
         [0, 0.5],
         [1, 0.5],
         [0, 0.75],
         [1, 0.75],
         [0, 1],
         [1, 1],
         [0.5, 1],
         [0.5, 0]
    ]
    for point in raw_points:
        points.InsertNextPoint([ point[1], point[0], 1.0 ])

    p1 = [
        [0, 2, 3],
        [0, 3, 1]
    ]
    p2 = [
        [2, 4, 5],
        [2, 5, 3]
    ]
    p3 = [
        [4, 6, 7],
        [4, 7, 5]
    ]
    p4 = [
        [6, 8, 9],
        [6, 9, 7]
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
