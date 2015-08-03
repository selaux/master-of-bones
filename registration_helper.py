import sys
import signal
import os
from PyQt4 import QtCore, QtGui

import helpers.loading as lh
import helpers.geometry as gh
import helpers.display as dh
import algorithms.registration as ar
import algorithms.triangulation as tr

BASE_PATH = os.getcwd()
print("Current base path: {0}".format(BASE_PATH))

ESTIMATORS = [
    {
        'label': 'Procrustes',
        'fn': ar.procrustes
    },
    {
        'label': 'Affine',
        'fn': ar.affine
    },
    {
        'label': 'Similarity',
        'fn': ar.similarity
    },
    {
        'label': 'Projective',
        'fn': ar.projective
    }
]
REFERENCE_ESTIMATORS = [
    {
        'label': 'Extract landmarks by angle',
        'fn': ar.get_marker_using_angles_estimator
    },
    {
        'label': 'Extract landmarks using space partitioning',
        'fn': ar.get_marker_using_space_partitioning_estimator
    },
    {
        'label': 'Use manually set markers',
        'fn': ar.get_manual_markers
    },
    {
        'label': 'Use manually set markers (5-9 only)',
        'fn': ar.get_manual_markers_5_to_9
    },
    {
        'label': 'Evaluated Spline Points',
        'fn': ar.get_spline_points_estimator
    },
    {
        'label': 'Nearest neighbors',
        'fn': ar.get_nearest_neighbor_point_estimator
    }
]

def normalize_outline(outline):
    triangulation = tr.triangulation(outline['bone_pixels'])

    outline['points'], outline['edges'] = gh.extract_outline(triangulation.points, triangulation.simplices)
    gh.normalize_outline_with_markers(outline)
    return outline

def open_directory_in_registration_helper():
    directory = QtGui.QFileDialog.getExistingDirectory(
        None,
        'Open Directory for Registration',
        BASE_PATH,
        QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks
    )

    if len(directory) == 0:
        QtCore.QCoreApplication.instance().quit()
        return False
    else:
        print("Directory to open: {}".format(str(directory)))
        loaded = lh.load_files(str(directory))
        loaded = map(normalize_outline, loaded)

        dh.registration(
            loaded,
            register_fn=ar.estimate_transform,
            estimators=ESTIMATORS,
            reference_estimators=REFERENCE_ESTIMATORS
        )

        return True

def main():
    def shutdown(signal, widget):
        exit()
    signal.signal(signal.SIGINT, shutdown)
    app = QtGui.QApplication(sys.argv)

    timer = QtCore.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    if open_directory_in_registration_helper():
        sys.exit(app.exec_())

if __name__ == '__main__':
    main()
