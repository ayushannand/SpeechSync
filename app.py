from functions.transcribe_retarded import transcribe_audio
from functions.translator import traslatorModel
from functions.lineBreak import getLines
from functions.saveasFile import save
from functions.postProcessHindi import postpHindi
from functions.ScareBleu import getScore


#Get and save score in 
refPath = "testing/ref.txt"
outPath = "testing/out.txt"
scoreName= "ta10, fbtranslate, = "
comments = ""
Stringify = False
print(getScore(refPath,outPath,scoreName,Stringify,comments))

