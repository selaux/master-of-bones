import sys
import signal
from PyQt4 import QtCore, QtGui

import helpers.display as dh

def main():
    def shutdown(signal, widget):
        exit()
    signal.signal(signal.SIGINT, shutdown)

    app = QtGui.QApplication(sys.argv)

    timer = QtCore.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    dh.synthetic_model_generation()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
