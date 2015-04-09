import itertools
import numpy as np
from skimage.color import label2rgb
from skimage.transform import resize
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import colorConverter
from matplotlib.collections import LineCollection

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

def outline(plt, outline, show_direction=False):
    fig, axes = plt.subplots()
    plt.axis('equal')
    axes.set_ylim(-1.5, 1.5)
    line = None

    lines = []
    colors = []
    for i, edge in enumerate(outline['edges']):
        color = colorConverter.to_rgb('r')
        if show_direction:
            color = (
                color[0] * i / len(outline['edges']),
                color[1] * i / len(outline['edges']),
                color[2] * i / len(outline['edges'])
            )
            
        colors.push(color)
        lines.push((
            outline['points'][edge[0], [1, 0]],
            outline['points'][edge[1], [1, 0]]
        ))
    
    collection = LineCollection(
        lines,
        colors=colors,
        linewidths=2,
        label=outline['label']
    )
    axes.add_collection(collection)
    axes.legend(handles=[collection])

    return fig, axes

def outlines(plt, outlines, show_direction=False):
    fig, axes = plt.subplots()
    plt.axis('equal')
    axes.set_ylim(-1.5, 1.5)
    colors = [
        'r',
        'g',
        'b',
        'c',
        'm',
        'y',
        'k'
    ]
    line_styles = [
       'solid',
        'dashed',
        'dashdot',
        'dotted'
    ]
    
    styles = list(itertools.product(line_styles, colors))
    handles = []

    collections = []
    for i, outline in enumerate(outlines):
        style = styles[i % len(styles)]
        line = None

        lines = []
        colors = []
        for i, edge in enumerate(outline['edges']):
            color = colorConverter.to_rgb(style[1])
            if show_direction:
                color = (
                    color[0] - color[0] * i / len(outline['edges']),
                    color[1] - color[1] * i / len(outline['edges']),
                    color[2] - color[2] * i / len(outline['edges'])
                )

            colors.append(color)
            lines.append((
                    outline['points'][edge[0], [1, 0]],
                    outline['points'][edge[1], [1, 0]]
                ))
            
        collection = LineCollection(
            lines,
            colors=colors,
            linewidths=2,
            linestyles=style[0],
            label=outline['label']
        )
        collections.append(collection)
        
    for collection in collections:
        axes.add_collection(collection)
    axes.legend(handles=collections)

    return fig, axes

def triangulation(plt, image, bone_pixels, recalc):
    tri = recalc(bone_pixels)
    fig, axes = plt.subplots()
    fig.subplots_adjust(bottom=0.1)

    def setup_triangulation(ax, tri):
        ax.set_ylim(image.shape[0], 0)
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
