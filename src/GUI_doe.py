#!/usr/bin/env python
"""
Module implementing GUI DoE.
"""
# Import PyQt Widgets for PyQt5 version
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem
# Import pyqtSlot to connect sliders and DoubleSpinBox signals
from PyQt5.QtCore import pyqtSlot
# Import Ui_MainWindow class from UiMainApp.py generated by uic module
from Uidoe import Ui_Design
# Import functions from numpy library for scientific simulation
from numpy import pi, linspace, meshgrid, sin
import numpy as np
# Import matplotlib.cm for the color map in our image of diffraction
import matplotlib.cm as cm
# Import design_of_experiments
import design_of_experiments as doe

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
        self.y = list()
        # It sets up layout and widgets that are defined
        # self.fig1()

    
    def fig1(self):
        lamda = 500*1.E-9 # self.slider_lambda.value()*1.E-9
        k = (2.*pi)/lamda  # wavelength of light in vaccuum
        b = 4*1e-5#self.slider_b.value()*1.E-5
        h = 4*1e-5#self.slider_h.value()*1.E-5
        # dimensions of diffracting rectangular aperture (m)
        #b is along (Ox) and h is along (Oy)
        f_2 = 2 #self.slider_f2.value()  # f2 is the focal length of the lens L2 (m)
        a = 15*1e-2#self.slider_a.value() * 1.E-2  # Side of a square-shaped screen (m)
        X_Mmax = a/2.
        X_Mmin = -a/2.
        Y_Mmax = X_Mmax
        Y_Mmin = X_Mmin
        N = 400
        X = linspace(X_Mmin, X_Mmax, N)
        Y = X  # coordinates of screen
        B = (k*b*X)/(2.*f_2)
        H = (k*h*Y)/(2.*f_2)  # intermediate variable
        # 2D representation
        BB, HH = meshgrid(B, H)
        I = ((sin(BB)/BB)**2)*((sin(HH)/HH)**2)
        # figure 2D
        mpl = self.pareto_fig.canvas
        mpl.ax.clear()
        mpl.ax.imshow(I, cmap=cm.gray, interpolation="bilinear",
                      origin="lower", vmin=0, vmax=.01)
        mpl.ax.set_xlabel(u"$X (m)$", fontsize=12, fontweight="bold")
        mpl.ax.set_ylabel(u"$Y (m)$", fontsize=12, fontweight="bold")
        mpl.ax.set_xticks(linspace(0, N, 5))
        mpl.ax.set_xticklabels(linspace(X_Mmin, X_Mmax, 5), color="r")
        mpl.ax.set_yticks(linspace(0, N, 5))
        mpl.ax.set_yticklabels(linspace(Y_Mmin, Y_Mmax, 5), color="r")
        mpl.figure.suptitle(
            "Fraunhofer Diffraction by rectangularaperture", fontsize=14, fontweight="bold")
        mpl.ax.set_title(r"$\lambda = %.3e \ m, \ b = %.2e \ m, \ h= %.2e \ m, \ f_2 = %.1f \ m$" % (
            lamda, b, h, f_2), fontsize=10)
        mpl.draw()


    # DoubleSpinBox signals
    @pyqtSlot("double")
    def on_n_parameters_valueChanged(self, value):
        self.tableWidget.setRowCount(2**value)
        self.tableWidget.setColumnCount(value)

        for i in range(int(value) + 1):
            self.tableWidget.setHorizontalHeaderItem(i, QTableWidgetItem('x_' + str(i) + ''))

        design = doe.gen_design(int(value))
        for x in range(2**int(value)):
            for y in range(int(value)):
                if design[x, y] == -1:
                    self.tableWidget.setItem(x, y, QTableWidgetItem('-'))
                elif design[x, y] == 1:
                    self.tableWidget.setItem(x, y, QTableWidgetItem('+'))

        self.measure.setRowCount(2**value)

    @pyqtSlot(QTableWidgetItem)
    def on_measure_itemChanged(self, item):
        n = self.measure.rowCount()
        #print(item.column(), item.row())

        if(item != None):
            try:
                float(item.text())
            except ValueError:
                self.measure.setItem(item.row(), item.column(), None)

        self.y = []
        for i in range(n):
            if self.measure.item(i,0) != None:
                self.y.append(float(self.measure.item(i,0).text()))
            else:
                return

        coef = np.dot(doe.gen_X_hat(n=int(np.log2(len(self.y)))), np.array(self.y))
        
        doe.draw_coefficents(self.coef_fig.canvas, coef, color="orange", title="")
        doe.draw_pareto(self.pareto_fig.canvas, coef, color="orange", title="")
        doe.draw_henry(self.henry_fig.canvas, coef, color="orange", title="")

        print(coef)

    """
    @pyqtSlot("double")
    def on_SpinBox_lambda_valueChanged(self, value):
        self.slider_lambda.setValue(value)

    @pyqtSlot("double")
    def on_SpinBox_b_valueChanged(self, value):
        self.slider_b.setValue(value)

    @pyqtSlot("double")
    def on_SpinBox_h_valueChanged(self, value):
        self.slider_h.setValue(value)

    @pyqtSlot("double")
    def on_SpinBox_a_valueChanged(self, value):
        self.slider_a.setValue(value)

    @pyqtSlot("double")
    def on_SpinBox_f2_valueChanged(self, value):
        self.slider_f2.setValue(value)

    # Sliders signals
    @pyqtSlot("int")
    def on_slider_lambda_valueChanged(self, value):
        self.SpinBox_lambda.setValue(value)
        self.fig1()

    @pyqtSlot("int")
    def on_slider_b_valueChanged(self, value):
        self.SpinBox_b.setValue(value)
        self.fig1()

    @pyqtSlot("int")
    def on_slider_h_valueChanged(self, value):
        self.SpinBox_h.setValue(value)
        self.fig1()

    @pyqtSlot("int")
    def on_slider_a_valueChanged(self, value):
        self.SpinBox_a.setValue(value)
        self.fig1()

    @pyqtSlot("int")
    def on_slider_f2_valueChanged(self, value):
        self.SpinBox_f2.setValue(value)
        self.fig1()
    """


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MyApplication = MainApp()
    MyApplication.show()  # Show the form
    sys.exit(app.exec_())  # Execute the app