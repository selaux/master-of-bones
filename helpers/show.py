# coding=utf-8

from functools import partial
from PyQt4 import QtCore


def show_window(window):
    app = QtCore.QCoreApplication.instance()
    if not hasattr(app, 'references') or not isinstance(app.references, set):
        app.references = set()
    app.references.add(window)
    def remove(app, window, event):
        app.references.remove(window)
    window.connect(window, QtCore.SIGNAL('triggered()'), partial(remove, app, window))
