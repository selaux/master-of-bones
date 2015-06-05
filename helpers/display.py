import itertools
import windows
from skimage.color import label2rgb
from skimage.transform import resize
from to_vtk import get_outline_actor, create_single_legend_actor
from show import show_window
from matplotlib.colors import colorConverter

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


def outline(outline, show_direction=False, title=None):
    window = windows.VTKWindow(title=title)
    show_window(window)
    base_color = (1.0, 0.0, 0.0)
    line_style = 0xFFFF

    window.render_actors([
        get_outline_actor(outline, base_color, line_style, show_direction)
    ])

    return window


def outlines(outlines, show_direction=False, title=None, color_by_class=False):
    window = windows.VTKWindow(title=title)
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


def triangulation(bones, do_triangulation):
    window = windows.TriangulationWindow(bones, do_triangulation)
    show_window(window)
    return window


def registration(bones, register_fn, estimators, reference_estimators):
    window = windows.RegistrationWindow(bones, register_fn, estimators, reference_estimators)
    show_window(window)
    return window

def comparison(bones, compare_fn, window_extractors, feature_functions):
    window = windows.ComparisonWindow(bones, compare_fn, window_extractors, feature_functions)
    show_window(window)
    return window

def synthetic_model_generation():
    window = windows.SyntheticGenerationWindow()
    show_window(window)
    return window
