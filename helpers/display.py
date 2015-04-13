import itertools
import numpy as np
import sys
from skimage.color import label2rgb
from skimage.transform import resize
import vtk
from windows import show_window, create_single_legend_actor, VTKWindow
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
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(image.shape[1], image.shape[0], 1)
    imageData.AllocateScalars(vtk.util.vtkConstants.VTK_UNSIGNED_CHAR, 3)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            imageData.SetScalarComponentFromFloat(x, y, 0, 0, image[y,x,0])
            imageData.SetScalarComponentFromFloat(x, y, 0, 1, image[y,x,1])
            imageData.SetScalarComponentFromFloat(x, y, 0, 2, image[y,x,2])
    return imageData

def to_vtk_image(image):
    copied = image.copy().astype(np.int16)
    #copied = copied.swapaxes(0,2)
    copied = copied.data
    imageImporter = vtk.vtkImageImport()
    #imageImporter.SetDataScalarTypeToUnsignedChar()
    imageImporter.SetDataExtent(0, image.shape[1]-1, 0, image.shape[0]-1, 0, 0)
    imageImporter.SetWholeExtent(0, image.shape[1]-1, 0, image.shape[0]-1, 0, 0)
    imageImporter.SetNumberOfScalarComponents(3)
    imageImporter.CopyImportVoidPointer(copied, len(copied))
    imageImporter.Update()
    return imageImporter.GetOutput()

def triangulation_vtk(image, bone_pixels, recalc):
    def create_data():
        tri = recalc(bone_pixels)

        points = vtk.vtkPoints()
        for point in tri.points:
            points.InsertNextPoint([ point[1], point[0], 1.0 ])

        triangles = vtk.vtkCellArray()
        for sim in tri.simplices:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0,sim[0])
            triangle.GetPointIds().SetId(1,sim[1])
            triangle.GetPointIds().SetId(2,sim[2])
            triangles.InsertNextCell(triangle)

        return points, triangles
        trianglePolyData.SetPoints(points)
        trianglePolyData.SetPolys(triangles)

    def rerender():
        pts, tri = create_data()
        trianglePolyData.SetPoints(pts)
        trianglePolyData.SetPolys(tri)
        trianglePolyData.Modified()
        window.vtkWidget.GetRenderWindow().Render()

    def on_key(obj, ev):
        key = obj.GetKeyCode()
        keymap = {
            'y': 255,
            'x': 0
        }

        try:
            if key and key in keymap:
                mouse_position = obj.GetEventPosition()
                print(mouse_position)
                if mouse_position:
                    window.ren.SetDisplayPoint((mouse_position[0], mouse_position[1], 0))
                    window.ren.DisplayToWorld()
                    x,y,z,w = window.ren.GetWorldPoint()

                    set_to = keymap[key]
                    if set_to == 0:
                        # Delete closest non zero pixel
                        indices_of_bone_pixels = np.nonzero(bone_pixels)
                        indices_of_bone_pixels = np.vstack(indices_of_bone_pixels).transpose()
                        indices_of_bone_pixels[:,0] = bone_pixels.shape[0] - indices_of_bone_pixels[:,0]
                        index_of_click = np.tile(np.array([ y, x ]), (indices_of_bone_pixels.shape[0], 1))
                        distances = np.linalg.norm(indices_of_bone_pixels - index_of_click, axis=1)

                        index_to_set = distances.argmin()
                        y = bone_pixels.shape[0] - indices_of_bone_pixels[index_to_set][0]
                        x = indices_of_bone_pixels[index_to_set][1]
                    else:
                        y = bone_pixels.shape[0] - y

                    bone_pixels[int(y), int(x)] = set_to
                    rerender()
        except Exception, e:
            print(e)

    def pick_event(obj, ev):
        print(obj, ev)
        #bone_pixels[eclick.ydata:erelease.ydata, eclick.xdata:erelease.xdata] = 0
        #bone_pixels[eclick.ydata:erelease.ydata:25, eclick.xdata:erelease.xdata:25] = 255
        #rerender()

    window = VTKWindow(title='Triangulation')
    show_window(window)

    imageActor = vtk.vtkImageActor()
    imageActor.SetInputData(to_vtk_image(image))

    tri = recalc(bone_pixels)

    trianglePolyData = vtk.vtkPolyData()
    pts, tri = create_data()
    trianglePolyData.SetPoints(pts)
    trianglePolyData.SetPolys(tri)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(trianglePolyData)

    triangulationActor = vtk.vtkActor()
    triangulationActor.SetMapper(mapper)
    triangulationActor.GetProperty().SetRepresentationToWireframe()
    triangulationActor.GetProperty().SetLineWidth(1.5)

    window.render_actors([ imageActor, triangulationActor ])

    window.areaPicker = vtk.vtkAreaPicker()
    window.iren.SetPicker(window.areaPicker)
    window.iren.AddObserver('KeyPressEvent', on_key, 0.0)
    window.areaPicker.AddObserver('EndPickEvent', pick_event, 0.0)

    return window


def triangulation(plt, image, bone_pixels, recalc):
    triangulation_vtk(image, bone_pixels, recalc)

    tri = recalc(bone_pixels)
    fig, axes = plt.subplots()
    fig.subplots_adjust(bottom=0.1)

    def setup_triangulation(ax, tri):
        ax.set_ylim(0, image.shape[0])
        ax.set_xlim(0, image.shape[1])
        ax.triplot(tri.points[:, 1], tri.points[:, 0], tri.simplices.copy(), 'c-')

    def rerender():
        while len(axes.lines) != 0:
            axes.lines[-1].remove()
        tri = recalc(bone_pixels)
        axes.triplot(tri.points[:, 1], tri.points[:, 0], tri.simplices.copy(), 'c-')
        fig.canvas.draw()

    def on_key(event):
        keymap = {
            'y': 255,
            'x': 0
        }
        if event.inaxes and event.key in keymap:
            x = int(event.xdata)
            y = int(event.ydata)
            set_to = keymap[event.key]
            if set_to == 0:
                # Delete closest non zero pixel
                indices_of_bone_pixels = np.nonzero(bone_pixels)
                indices_of_bone_pixels = np.vstack(indices_of_bone_pixels).transpose()
                index_of_click = np.tile(np.array([ y, x ]), (indices_of_bone_pixels.shape[0], 1))
                distances = np.linalg.norm(indices_of_bone_pixels - index_of_click, axis=1)

                index_to_set = distances.argmin()
                y = indices_of_bone_pixels[index_to_set][0]
                x = indices_of_bone_pixels[index_to_set][1]

            bone_pixels[y, x] = set_to
            rerender()
        if event.key == 'a':
            fig.fill_area_selector.set_active(False)
            fig.remove_area_selector.set_active(False)
            fig.fill_area_selector.update()
            fig.remove_area_selector.update()

    def key_release(event):
        if event.key == 'a':
            fig.fill_area_selector.set_active(True)
            fig.remove_area_selector.set_active(True)

    def remove_pixels_from_area(eclick, erelease):
        bone_pixels[eclick.ydata:erelease.ydata, eclick.xdata:erelease.xdata] = 0
        rerender()

    def fill_area_with_pixels(eclick, erelease):
        bone_pixels[eclick.ydata:erelease.ydata, eclick.xdata:erelease.xdata] = 0
        bone_pixels[eclick.ydata:erelease.ydata:25, eclick.xdata:erelease.xdata:25] = 255
        rerender()

    #axes = plt.axes([0.1, 0.1, 0.8, 0.8])
    axes.imshow(image)
    setup_triangulation(axes, tri)

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('key_release_event', key_release)
    fig.remove_area_selector = RectangleSelector(
        axes,
        remove_pixels_from_area,
        drawtype='box',
        button=[1],
        minspanx=20,
        minspany=20,
        spancoords='pixels',
        rectprops = dict(edgecolor='red', fill=False)
    )
    fig.remove_area_selector.set_active(True)
    fig.fill_area_selector = RectangleSelector(
        axes,
        fill_area_with_pixels,
        drawtype='box',
        button=[3],
        minspanx=20,
        minspany=20,
        spancoords='pixels',
        rectprops = dict(edgecolor='green', fill=False)
    )
    fig.fill_area_selector.set_active(True)

    return fig, axes
