import sys
from PyQt4 import QtGui

import helpers.display as dh

def main():
    app = QtGui.QApplication(sys.argv)
    dh.synthetic_model_generation()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
