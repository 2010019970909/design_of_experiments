# https://stackoverflow.com/questions/32035251/displaying-latex-in-pyqt-pyside-qtablewidget
import sys
import matplotlib as mpl
from functools import partial
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5 import QtGui, QtCore, QtWidgets


def mathTex_to_QPixmap(mathTex, fs):

    # ---- set up a mpl figure instance ----
    fig = Figure()
    fig.patch.set_facecolor('none')
    fig.set_canvas(FigureCanvasAgg(fig))
    renderer = fig.canvas.get_renderer()

    # ---- plot the mathTex expression ----

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.patch.set_facecolor('none')
    t = ax.text(0, 0, mathTex, ha='left', va='bottom', fontsize=fs)

    # ---- fit figure size to text artist ----

    fwidth, fheight = fig.get_size_inches()
    fig_bbox = fig.get_window_extent(renderer)

    text_bbox = t.get_window_extent(renderer)

    tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
    tight_fheight = text_bbox.height * fheight / fig_bbox.height

    fig.set_size_inches(tight_fwidth, tight_fheight)

    # ---- convert mpl figure to QPixmap ----

    buf, size = fig.canvas.print_to_buffer()
    qimage = QtGui.QImage.rgbSwapped(QtGui.QImage(buf, size[0], size[1],
                                                  QtGui.QImage.Format_ARGB32))
    qpixmap = QtGui.QPixmap(qimage)

    return qpixmap


class LaTeXQTableWidget(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super(LaTeXQTableWidget, self).__init__(parent)

        self.setHorizontalHeader(MyHorizHeader(self))
        """
        # https://stackoverflow.com/questions/51620130/have-enter-key-behave-like-tab-in-qtablewidget
        for key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            QtWidgets.QShortcut(key, self,
                                partial(self.focusNextPrevChild, True))
        """    
    """
    def keyPressEvent(self, ev):        
        if ev.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            print("Enter key pressed")
            self.focusNextPrevChild(True)
    """

    def setHorizontalHeaderLabels(self, headerLabels, fontsize):

        qpixmaps = []
        indx = 0
        for labels in headerLabels:
            qpixmaps.append(mathTex_to_QPixmap(labels, fontsize))
            self.setColumnWidth(indx, qpixmaps[indx].size().width() + 16)
            indx += 1

        self.horizontalHeader().qpixmaps = qpixmaps

        super(LaTeXQTableWidget, self).setHorizontalHeaderLabels(headerLabels)


class MyHorizHeader(QtWidgets.QHeaderView):
    def __init__(self, parent):
        super(MyHorizHeader, self).__init__(QtCore.Qt.Horizontal, parent)

        self.setSectionsClickable(True)
        self.setStretchLastSection(True)

        self.qpixmaps = []

    def paintSection(self, painter, rect, logicalIndex):

        if not rect.isValid():
            return

        # ------------------------------ paint section (without the label) ----

        opt = QtWidgets.QStyleOptionHeader()
        self.initStyleOption(opt)

        opt.rect = rect
        opt.section = logicalIndex
        opt.text = ""

        # ---- mouse over highlight ----

        mouse_pos = self.mapFromGlobal(QtGui.QCursor.pos())
        if rect.contains(mouse_pos):
            opt.state |= QtWidgets.QStyle.State_MouseOver

        # ---- paint ----

        painter.save()
        self.style().drawControl(QtWidgets.QStyle.CE_Header, opt, painter, self)
        painter.restore()

        # ------------------------------------------- paint mathText label ----
        if len(self.qpixmaps) == 0:
            return

        elif len(self.qpixmaps) <= logicalIndex:
            return

        # print(len(self.qpixmaps), logicalIndex)
        qpixmap = self.qpixmaps[logicalIndex]

        # ---- centering ----

        xpix = int((rect.width() - qpixmap.size().width()) / 2. + rect.x())
        ypix = int((rect.height() - qpixmap.size().height()) / 2.)

        # ---- paint ----

        rect = QtCore.QRect(xpix, ypix, qpixmap.size().width(),
                            qpixmap.size().height())
        painter.drawPixmap(rect, qpixmap)

    def sizeHint(self):

        baseSize = QtWidgets.QHeaderView.sizeHint(self)

        baseHeight = baseSize.height()
        if len(self.qpixmaps):
            for pixmap in self.qpixmaps:
                baseHeight = max(pixmap.height() + 8, baseHeight)
        baseSize.setHeight(baseHeight)

        self.parentWidget().repaint()

        return baseSize


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    w = LaTeXQTableWidget()
    w.verticalHeader().hide()

    headerLabels = [
        '$C_{soil}=(1 - n) C_m + \\theta_w C_w$',
        '$k_{soil}=\\frac{\\sum f_j k_j \\theta_j}{\\sum f_j \\theta_j}$',
        '$\\lambda_{soil}=k_{soil} / C_{soil}$']

    w.setColumnCount(len(headerLabels))
    w.setHorizontalHeaderLabels(headerLabels, 25)
    w.setRowCount(3)
    w.setAlternatingRowColors(True)

    k = 1
    for j in range(3):
        for i in range(3):
            w.setItem(i, j, QtWidgets.QTableWidgetItem('Value %i' % (k)))
            k += 1

    w.show()
    w.resize(700, 200)

    sys.exit(app.exec_())
