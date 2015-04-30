import itertools
import numpy as np
import sys
from time import time
from skimage.color import label2rgb
from skimage.transform import resize
import vtk
from windows import show_window, create_single_legend_actor, VTKWindow, TriangulationWindow, RegistrationWindow
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import colorConverter
from matplotlib.collections import LineCollection
from vtk.util.vtkConstants import *

ALL_COLORS = [
    'r',
    'g',
    'b',
    'c',
    'm',
    'y',
    'k'
]
ALL_LINE_STYLES = [
   0xFFFF,
   0xF0F0,
   0xFF00,
   0x000F
]
ALL_STYLES = list(itertools.product(ALL_LINE_STYLES, ALL_COLORS))

def image(plt, image):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    ax.set_title("Image")

def clusters(plt, image, clusters):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].set_title("RGB")
    ax[1].imshow(resize(label2rgb(clusters), image.shape))
    ax[1].set_title("Clusters")

def feature(plt, image, f, index=0):
    reshaped = f.reshape((image.shape[0], image.shape[1]))
    fig, ax = plt.subplots(1)
    ax.imshow(reshaped)
    ax.set_title("Feature {0}".format(index))

def features(plt, image, features):
    number_of_features = features.shape[1] if len(features.shape) > 1 else 1

    if number_of_features != 1:
        for i in range(features.shape[1]):
            feature(plt, image, features[:, i], i)
    else:
        feature(plt, image, features)

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

def outline(outline, show_direction=False, title=None):
    window = VTKWindow(title=title)
    show_window(window)
    base_color = (1.0, 0.0, 0.0)
    line_style = 0xFFFF

    window.render_actors([
        get_outline_actor(outline, base_color, line_style, show_direction)
    ])

    return window

def outlines(outlines, show_direction=False, title=None, color_by_class=False):
    window = VTKWindow(title=title)
    show_window(window)

    actors = []
    legends = []
    for i, outline in enumerate(outlines):
        if color_by_class:
            base_color = ALL_COLORS[(outline['class']-1) % len(ALL_STYLES)]
            line_style = ALL_LINE_STYLES[i % len(ALL_LINE_STYLES)]
            style = [ line_style, base_color ]
        else:
            style = ALL_STYLES[i % len(ALL_STYLES)]
        base_color = colorConverter.to_rgb(style[1])
        line_style = style[0]

        actors.append(get_outline_actor(outline, base_color, line_style, show_direction))
        legends.append((outline['label'], base_color, create_single_legend_actor(base_color, line_style)))
    window.render_actors(actors, legends)

    return window

def to_vtk_image(image):
    image = np.flipud(image.copy()).astype('uint8')
    print(image)
    print(image.shape)

    start = time()
    imageString = image.tostring()
    imageImporter = vtk.vtkImageImport()
    imageImporter.CopyImportVoidPointer(imageString, len(imageString))
    imageImporter.SetDataScalarTypeToUnsignedChar()
    imageImporter.SetNumberOfScalarComponents(3)
    imageImporter.SetDataExtent(0,image.shape[1]-1,
                                0,image.shape[0]-1,
                                0,0)
    imageImporter.SetWholeExtent(0,image.shape[1]-1,
                                 0,image.shape[0]-1,
                                 0,0)
    imageImporter.Update()
    interval = time() - start
    print(interval)

    return imageImporter

def triangulation(bones, do_triangulation):
    window = TriangulationWindow(bones, do_triangulation)
    show_window(window)

def registration(bones, register_fn, estimators, reference_estimators):
    window = RegistrationWindow(bones, register_fn, estimators, reference_estimators)
    show_window(window)