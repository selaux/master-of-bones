from functools import partial
import sys
import signal
import os
import io
from skimage.data import imread
from PyQt4 import QtCore, QtGui

import helpers.display as dh
import helpers.loading as lh
import algorithms.triangulation as tr
import algorithms.segmentation as se

BASE_PATH = os.getcwd()
DATA_PATH = os.path.join(BASE_PATH, 'data/2D/triangulation-results')
SEGMENTATION_METHODS = [
    {
        'label': 'Watershed',
        'fn': se.watershed
    },
    {
        'label': 'SLIC',
        'fn': se.slic
    },
    {
        'label': 'Felzenszwalb',
        'fn': se.felzenszwalb
    },
    {
        'label': 'Active Contours',
        'fn': se.active_contours
    },
    {
        'label': 'Gabor Filters',
        'fn': se.gabor
    },
    {
        'label': 'Textural Edges',
        'fn': se.textural_edges
    }
]


print("Current base path: {0}".format(BASE_PATH))

def main():
    def shutdown(signal, widget):
        exit()
    signal.signal(signal.SIGINT, shutdown)
    app = QtGui.QApplication(sys.argv)

    timer = QtCore.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    loaded = lh.load_files(str(DATA_PATH), filter_unfinished=False)
    for l in loaded:
        jpg_io = io.BytesIO(l['jpg_image'])
        l['image'] = imread(jpg_io)
        l['save_path'] = os.path.join(DATA_PATH, l['filename'])
        if 'edited' not in l:
            l['edited'] = False

    dh.triangulation(
        loaded,
        do_triangulation=tr.triangulation,
        do_import=partial(dh.import_bone, SEGMENTATION_METHODS)
    )

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
