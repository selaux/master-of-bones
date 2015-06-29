import sys
import signal
import os
from PyQt4 import QtCore, QtGui

import helpers.display as dh
import helpers.loading as lh
import algorithms.comparison as cr

BASE_PATH = os.getcwd()
print("Current base path: {0}".format(BASE_PATH))

FEATURE_EXTRACTORS = [
    {
        'label': 'Spline Points (Flattened)',
        'fn': cr.FeatureFlattenSplines
    },
    {
        'label': 'Distances to center',
        'fn': cr.FeatureDistanceToCenter
    },
    {
        'label': 'Distances to center and curvature',
        'fn': cr.FeatureDistanceToCenterAndCurvature
    },
    {
        'label': 'Use derivation vector from mean bone',
        'fn': cr.FeatureFlattenedDeviationVectorFromMeanBone
    },
    {
        'label': 'Spline derivatives',
        'fn': cr.FeatureSplineDerivatives
    },
    {
        'label': 'Distances to Markers',
        'fn': cr.FeatureDistancesToMarkers
    },
    {
        'label': 'Use Curvature of Distance from Center',
        'fn': cr.FeatureCurvatureOfDistanceFromCenter
    }
]
WINDOW_EXTRACTORS = [
    {
        'label': '2D Window by length',
        'fn': cr.WindowExtractor2DByWindowLength
    }
]

def open_directory_in_comparison_helper():
    directory = QtGui.QFileDialog.getExistingDirectory(
        None,
        'Open Directory for Comparison',
        BASE_PATH,
        QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks
    )

    if len(directory) == 0:
        QtCore.QCoreApplication.instance().quit()
        return False
    else:
        print("Directory to open: {}".format(str(directory)))
        loaded = lh.load_files(str(directory))

        dh.comparison(
            loaded,
            compare_fn=cr.do_comparison,
            window_extractors=WINDOW_EXTRACTORS,
            feature_extractors=FEATURE_EXTRACTORS
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

    if open_directory_in_comparison_helper():
        sys.exit(app.exec_())

if __name__ == '__main__':
    main()
