import signal
import os
import sys
from PyQt4 import QtCore, QtGui

import algorithms.landmarks as la
import algorithms.measurable_differences as md
import helpers.loading as lh
import helpers.display as dh

BASE_PATH = os.getcwd()
print("Current base path: {0}".format(BASE_PATH))

LANDMARK_EXTRACTORS = [
    {
        'label': 'By Angle and Distance From Center',
        'fn': la.get_using_angles
    },
    {
        'label': 'By Space Partitioning an extremas in x/y direction',
        'fn': la.get_using_space_partitioning
    }
]

def open_directory_in_measurable_distances_helper():
    directory = QtGui.QFileDialog.getExistingDirectory(
        None,
        'Open Directory to find measurable differences',
        BASE_PATH,
        QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks
    )

    if len(directory) == 0:
        QtCore.QCoreApplication.instance().quit()
        return False
    else:
        print("Directory to open: {}".format(str(directory)))
        loaded = lh.load_files(str(directory))

        dh.measurable_differences(loaded, md.find_measurable_differences, landmark_extractors=LANDMARK_EXTRACTORS)

        return True

def main():
    def shutdown(signal, widget):
        exit()
    signal.signal(signal.SIGINT, shutdown)
    app = QtGui.QApplication(sys.argv)

    timer = QtCore.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    if open_directory_in_measurable_distances_helper():
        sys.exit(app.exec_())

if __name__ == '__main__':
    main()
