from functions.transcribe import transcribe_audio
from functions.translator import traslatorModel
from functions.lineBreak import getLines
from functions.saveasFile import save
from functions.postProcessHindi import postpHindi
from functions.ScareBleu import getScore


#Get and save score in 
refPath = "testing/tamilRef.txt"
outPath = "testing/ta10.txt"
scoreName= "ta10, fbtranslate, = "
comments = ""
Stringify = True
print(getScore(refPath,outPath,scoreName,Stringify,comments))

