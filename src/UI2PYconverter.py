# −∗− coding: utf−8 −∗−
''' PyQt5 uic module convert ui file (XML code) into py file (Python code) '''
from PyQt5 import uic

fin = open('doe.ui', 'r')
fout = open('Uidoe.py', 'w')
uic.compileUi(fin, fout, execute=True)
fin.close()
fout.close()
