from PyQt4 import QtCore, QtGui

class BoneList(QtGui.QListView):
    def __init__(self, bones, parent=None):
        QtGui.QListView.__init__(self, parent)

        lm = BoneListModel(bones, self)
        self.setModel(lm)
        self.selectionModel().select(self.model().index(0), QtGui.QItemSelectionModel.SelectCurrent)

class BoneListModel(QtCore.QAbstractListModel):
    def __init__(self, datain, parent=None):
        QtCore.QAbstractListModel.__init__(self, parent)
        self.listdata = datain

    def rowCount(self, parent):
        return len(self.listdata)

    def data(self, index, role):
        if role == QtCore.Qt.DecorationRole:
            row = index.row()
            edited = 'edited' in self.listdata[row] and self.listdata[row]['edited']
            done = 'done' in self.listdata[row] and self.listdata[row]['done']
            class_label = self.listdata[row]['class_short']

            color = QtGui.QColor(255, 0, 0)
            if done:
                color = QtGui.QColor(0, 255, 0)
            elif edited:
                color = QtGui.QColor(255, 140, 0)

            pixmap = QtGui.QPixmap(26, 26)
            painter = QtGui.QPainter()
            pixmap.fill(color)
            painter.begin(pixmap)
            painter.setFont(QtGui.QFont('Helvetica', 10))
            painter.drawText(QtCore.QRectF(0, 0, 26, 26), QtCore.Qt.AlignCenter, class_label.decode('utf-8'))
            painter.end()

            return QtGui.QIcon(pixmap)
        if role == QtCore.Qt.DisplayRole:
            row = index.row()
            return self.listdata[row]['filename'].decode('utf-8')
