from math import degrees, floor
import os
import vtk
import matplotlib.cm as cmx

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import cluster
import helpers.loading as lh
import helpers.geometry as gh
import algorithms.splines as sa
import algorithms.triangulation as ta

# we create 40 separable points
def generate_svm():
    np.random.seed(10)
    X = np.r_[np.random.randn(10, 2) - [2, 2], np.random.randn(10, 2) + [2, 2]]
    Y = [0] * 10 + [1] * 10

    # figure number
    fignum = 1

    # fit the model
    clf = svm.SVC(kernel='linear')
    clf.fit(X, Y)

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy + a * margin
    yy_up = yy - a * margin

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, 'k-', linewidth=3)
    plt.plot(xx, yy_down, 'k--', linewidth=3)
    plt.plot(xx, yy_up, 'k--', linewidth=3)

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=480,
                facecolors='none', zorder=10)
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, s=240, cmap=plt.cm.Paired)

    plt.show()

def generate_spline():
    pts = np.array([
        [ 0.0, 0.0 ],
        [ 1, 1.5 ],
        [ 0, 2 ],
        [ -1, 2.5 ],
        [ -2, 0 ],
        [ -1, -2.5],
        [ 0.0, -2.0],
        [ 1, -1.5]
    ])
    x = pts[:, 0]
    y = pts[:, 1]

    tck, u = sa.get_spline_params(pts)

    t = np.linspace(0.0, 1.0, num=100)
    new = sa.evaluate_spline(t, tck)
    xnew = new[:, 0]
    ynew = new[:, 1]

    plt.figure()
    plt.plot(x, y, 'o', xnew, ynew, linewidth=3, ms=10)
    plt.title('Cubic-spline interpolation')
    plt.show()

def generate_data_for_clustering():
    np.random.seed(4)
    f1 = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.01]], size=(15))
    f2 = np.random.multivariate_normal([1, 0], [[0.1, 0], [0, 0.01]], size=(15))
    f3 = np.random.multivariate_normal([0, 1], [[0.01, 0], [0, 0.01]], size=(15))
    f4 = np.random.multivariate_normal([1, 1], [[0.01, 0], [0, 0.15]], size=(15))

    print(f1.shape)
    return np.concatenate((f1, f2, f3, f4))

def draw_clustering(data, classes):
    color = cmx.Set1

    plt.figure()
    plt.plot(data[classes == 0, 0], data[classes == 0, 1], 'o', ms=20, color=color(0))
    plt.plot(data[classes == 1, 0], data[classes == 1, 1], 'o', ms=20, color=color(0.125))
    plt.plot(data[classes == 2, 0], data[classes == 2, 1], 'o', ms=20, color=color(0.25))
    plt.plot(data[classes == 3, 0], data[classes == 3, 1], 'o', ms=20, color=color(0.375))
    plt.show()

def generate_dbscan():
    data = generate_data_for_clustering()
    dbscan = cluster.DBSCAN(eps=0.5, min_samples=3)
    classes = dbscan.fit_predict(data)
    draw_clustering(data, classes)


def generate_kmeans():
    data = generate_data_for_clustering()
    kmeans = cluster.KMeans(n_clusters=4)
    classes = kmeans.fit_predict(data)
    draw_clustering(data, classes)

def get_axis_limits(metrics):
    max_value = max(metrics)
    min_value = min(metrics)
    padding = 0.1 * max_value

    max_value += padding
    min_value -= padding
    min_value = min_value if min_value >= 0 else 0

    max_value = round(max_value, 2)
    min_value = round(min_value, 2)

    return min_value, max_value

def generate_distance_plot():
    def add_spline(outline):
        outline['spline_params'] = sa.get_spline_params(outline['points'])[0]
        return outline

    bones = map(add_spline, lh.load_files('data/2D/registered-outline/2015-06-08'))
    class1bones = [ bone for bone in bones if bone['class'] == 2 ]
    class2bones = [ bone for bone in bones if bone['class'] == 3 ]
    class1outline = np.mean([sa.evaluate_spline(np.linspace(0, 1, 180, endpoint=False), s['spline_params']) for s in class1bones], axis=0)
    class2outline = np.mean([sa.evaluate_spline(np.linspace(0, 1, 180, endpoint=False), s['spline_params']) for s in class2bones], axis=0)
    total_mean = np.mean([sa.evaluate_spline(np.linspace(0, 1, 180, endpoint=False), s['spline_params']) for s in bones], axis=0)

    distances = np.linalg.norm(class1outline - class2outline, axis=1)
    angles = np.array([degrees(gh.cart2pol(ev[0], ev[1])[1]) for ev in total_mean])
    angles[angles[:] < 0] += 360

    limit_min, limit_max = get_axis_limits(distances)

    chart = vtk.vtkChartXY()
    chart.GetAxis(1).SetBehavior(vtk.vtkAxis.FIXED)
    chart.GetAxis(0).SetBehavior(vtk.vtkAxis.FIXED)
    chart.GetAxis(0).SetTitle('')
    chart.GetAxis(1).SetTitle('')
    chart.GetAxis(0).GetLabelProperties().SetFontSize(30)
    chart.GetAxis(1).GetLabelProperties().SetFontSize(30)
    chart.GetAxis(1).SetRange(0, 359)
    chart.GetAxis(0).SetRange(0, 1)
    chart_table = vtk.vtkTable()
    chart_angle_axis_array = vtk.vtkFloatArray()
    chart_angle_axis_array.SetName('Angle (degrees)')
    chart_metric_array = vtk.vtkFloatArray()
    chart_metric_array.SetName('Distance')
    chart_table.AddColumn(chart_angle_axis_array)
    chart_table.AddColumn(chart_metric_array)

    chart_line = chart.AddPlot(vtk.vtkChart.LINE)
    chart_line.SetColor(0, 0, 0, 255)
    chart_line.SetWidth(2)

    chart.GetAxis(0).SetGridVisible(False)
    chart.GetAxis(1).SetGridVisible(False)
    chart_line.SetWidth(5)

    chart_table.SetNumberOfRows(distances.shape[0])
    for i in range(distances.shape[0]):
        chart_table.SetValue(i, 0, angles[i])
        chart_table.SetValue(i, 1, distances[i])
    chart_line.SetInputData(chart_table, 0, 1)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(640, 360)

    tick_positions_x = vtk.vtkDoubleArray()
    tick_positions_y = vtk.vtkDoubleArray()
    for angle in [0, 90, 180, 270, 360]:
        tick_positions_x.InsertNextValue(angle)
    for height in [limit_min, floor(limit_max*100) / 100]:
        tick_positions_y.InsertNextValue(height)
    chart.GetAxis(0).SetRange(limit_min, limit_max)
    chart.GetAxis(0).SetCustomTickPositions(tick_positions_y)
    chart.GetAxis(1).SetRange(0, 360)
    chart.GetAxis(1).SetCustomTickPositions(tick_positions_x)
    chart.Update()

    graph_context = vtk.vtkContextView()
    graph_context.SetRenderWindow(render_window)

    graph_context.GetScene().AddItem(chart)

    exporter = vtk.vtkGL2PSExporter()
    exporter.SetRenderWindow(render_window)
    exporter.SetFileFormatToSVG()
    exporter.CompressOff()
    exporter.DrawBackgroundOff()
    exporter.SetFilePrefix(os.path.abspath('thesis/img/results/real/distances'))
    exporter.Write()

    render_window.Finalize()

#generate_svm()
#generate_spline()
generate_kmeans()
generate_dbscan()
# generate_distance_plot()
