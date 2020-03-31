#!/usr/bin/env python
"""
Module implementing GUI DoE.
"""
# Import PyQt Widgets for PyQt5 version
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QHeaderView
# Import pyqtSlot to connect sliders and DoubleSpinBox signals
from PyQt5.QtCore import pyqtSlot, Qt
# Import QIcon
from PyQt5.QtGui import QIcon
# Import Ui_MainWindow class from UiMainApp.py generated by uic module
from Uidoe import Ui_Design
# Import functions from numpy library for scientific simulation
from numpy import pi, linspace, meshgrid, sin
import numpy as np
# Import matplotlib.cm for the color map in our image of diffraction
import matplotlib.cm as cm
# Import design_of_experiments
import design_of_experiments as doe
# To add a key binding
from functools import partial

class MainApp(QMainWindow, Ui_Design):
    """
    MainApp class inherit from QMainWindow and from
    Ui_MainWindow class in UiMainApp module.
    """

    def __init__(self):
        """Constructor or the initializer"""
        QMainWindow.__init__(self)
        # It is imperative to call self.setupUi (self) for the interface to initialize.
        # This is defined in design.py file automatically
        self.setupUi(self)
        self.setWindowTitle("Design of Experiments - by Vincent STRAGIER")
        self.n_parameters.setValue(1)
        self.n_parameters.setMinimum(1)
        self.n_parameters.setMaximum(8)
        header = self.measure.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        self.tableWidget.setAlternatingRowColors(True)
        self.measure.setAlternatingRowColors(True)
        self.coefficients_tab.setAlternatingRowColors(True)
        # https://doc.qt.io/qt-5/qtabwidget.html#setTabEnabled
        self.tabWidget.setTabEnabled(1, False)
        self.tabWidget.setTabEnabled(2, False)
        self.tabWidget.setTabEnabled(3, False)
        self.y = list()
        # It sets up layout and widgets that are defined
        self.showMaximized()
        self.pushButton.setEnabled(False)
        self.pushButton.clicked.connect(self.reset_y)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        if e.key() == Qt.Key_F11:
            if self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()

    def reset_y(self):
        n = self.measure.rowCount()
        self.y = []

        for i in range(n):
            self.measure.setItem(i, 0, QTableWidgetItem(None))

    # DoubleSpinBox signals
    @pyqtSlot("double")
    def on_n_parameters_valueChanged(self, value):
        self.tableWidget.setRowCount(int(2**value))
        self.tableWidget.setColumnCount(int(value))
        self.measure.setRowCount(int(2**value))

        labels = doe.gen_a_labels(n=int(value))
        self.coefficients_tab.setColumnCount(len(labels))
        self.coefficients_tab.setHorizontalHeaderLabels(labels, 12)
        self.measure.setHorizontalHeaderLabels(['$y$'], 12)

        header = []
        for i in range(int(value)):
            header.append("$x_" + str(i) + "$")

        self.tableWidget.setHorizontalHeaderLabels(header, 12)

        design = doe.gen_design(int(value))
        for x in range(2**int(value)):
            for y in range(int(value)):
                if design[x, y] == -1:
                    item = QTableWidgetItem('-')  # create the item

                elif design[x, y] == 1:
                    item = QTableWidgetItem('+')  # create the item

                item.setTextAlignment(Qt.AlignHCenter)  # change the alignment
                self.tableWidget.setItem(x, y, item)

    @pyqtSlot(QTableWidgetItem)
    def on_measure_itemChanged(self, item):
        n = self.measure.rowCount()

        if(item != None):
            try:
                float(item.text())
            except ValueError:
                self.measure.setItem(item.row(), item.column(), None)

        self.y = []
        for i in range(n):
            if self.measure.item(i, 0) != None:
                self.y.append(float(self.measure.item(i, 0).text()))
            else:
                if len(self.y) == 0:
                    self.pushButton.setEnabled(False)
                else:
                    self.pushButton.setEnabled(True)

                self.tabWidget.setTabEnabled(1, False)
                self.tabWidget.setTabEnabled(2, False)
                self.tabWidget.setTabEnabled(3, False)
                return

        # Enable the tabs
        self.tabWidget.setTabEnabled(1, True)
        self.tabWidget.setTabEnabled(2, True)
        self.tabWidget.setTabEnabled(3, True)

        # Generate the table of coefficient
        coef = np.dot(doe.gen_X_hat(
            n=int(np.log2(len(self.y)))), np.array(self.y))
        labels = doe.gen_a_labels(n=int(np.log2(len(self.y))))

        self.coefficients_tab.setRowCount(1)
        self.coefficients_tab.setColumnCount(len(labels))
        self.coefficients_tab.setHorizontalHeaderLabels(labels, 12)

        for i in range(len(labels)):
            try:
                item = QTableWidgetItem(str(coef[i]))
                item.setTextAlignment(Qt.AlignHCenter)  # change the alignment
                self.coefficients_tab.setItem(0, i, item)
            except:
                print("error")

    @pyqtSlot(int)
    def on_tabWidget_currentChanged(self, index):
        """ https://doc.qt.io/qt-5/qtabwidget.html#currentChanged """
        if index == 1:
            # Generate the table of coefficient
            coef = np.dot(doe.gen_X_hat(
                n=int(np.log2(len(self.y)))), np.array(self.y))
            doe.clear_draw(self.coef_fig.canvas)
            doe.draw_coefficents(self.coef_fig.canvas,
                                 coef, color="blue", title="")

        elif index == 2:
            coef = np.dot(doe.gen_X_hat(
                n=int(np.log2(len(self.y)))), np.array(self.y))
            doe.clear_draw(self.pareto_fig.canvas)
            doe.draw_pareto(self.pareto_fig.canvas,
                            coef, color="blue", title="")

        elif index == 3:
            coef = np.dot(doe.gen_X_hat(
                n=int(np.log2(len(self.y)))), np.array(self.y))
            doe.clear_draw(self.henry_fig.canvas)
            doe.draw_henry(self.henry_fig.canvas, coef,
                           empirical_cumulative_distribution="modified", color="blue", title="")


if __name__ == "__main__":
    # For Windows set AppID to add an Icon in the taskbar
    # https://stackoverflow.com/questions/1551605/how-to-set-applications-taskbar-icon-in-windows-7
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        appid = u'vincent_stragier.umons.doe.v1.0.0'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)

        lpBuffer = wintypes.LPWSTR()
        AppUserModelID = ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID
        AppUserModelID(ctypes.cast(ctypes.byref(lpBuffer), wintypes.LPWSTR))
        appid = lpBuffer.value
        ctypes.windll.kernel32.LocalFree(lpBuffer)
        
    app = QApplication(sys.argv)
    MyApplication = MainApp()
    MyApplication.show()  # Show the form

    app.setWindowIcon(QIcon('fpms.svg'))
    MyApplication.setWindowIcon(QIcon('fpms.svg'))
    sys.exit(app.exec_())  # Execute the app
