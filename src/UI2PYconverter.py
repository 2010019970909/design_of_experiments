# −∗− coding: utf−8 −∗−
''' PyQt5 uic module convert ui file (XML code) into py file (Python code) '''
from PyQt5 import uic

with open('doe.ui', 'r') as fin:
    with open('Uidoe.py', 'w') as fout:
        uic.compileUi(fin, fout, execute=True)
