rem luatex  -interaction=nonstopmode -ini -jobname="pream" "&lualatex" mylatexformat.ltx --output-directory build %*.tex "./pream.tex"
lualatex -draftmode -interaction=nonstopmode --output-directory build %*.tex
"C:/Program Files (x86)/MiKTeX 2.9/biber/bin/windows-x86/biber.exe" --input-directory build --output-directory build %*
pythontex build/%*
lualatex -interaction=nonstopmode --output-directory build %*.tex
rem "C:/Program Files (x86)/Foxit Software/Foxit Reader/FoxitReader.exe" build/%*.pdf
rem "C:/Program Files (x86)/Adobe/Acrobat Reader DC/Reader/AcroRd32.exe" build/%*.pdf
rem lualatex -interaction=nonstopmode --output-directory build %.tex|
rem "C:/Program Files (x86)/MiKTeX 2.9/biber/bin/windows-x86/biber.exe" --input-directory build --output-directory build %|
rem pythontex build/%|
rem lualatex -interaction=nonstopmode --output-directory build %.tex|lualatex -interaction=nonstopmode --output-directory build %.tex|
rem "C:/Program Files/Adobe/Reader 11.0/Reader/AcroRd32.exe" %.pdf