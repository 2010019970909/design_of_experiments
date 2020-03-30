# <python> -m install PySide2
import sys
from PySide2.QtWidgets import QApplication, QLabel

app = QApplication(sys.argv) # or app = QApplication([])
# label = QLabel("Hello World!") or
label = QLabel("<font color=red size=40>Hello World!</font>")
label.show()
app.exec_()
