from functions.transcribe import transcribe_audio
from functions.translator import traslatorModel
from functions.lineBreak import getLines
from functions.saveasFile import save
from functions.postProcessHindi import postpHindi
from functions.ScareBleu import getScore


#Get and save score in 
refPath = "testing/Hindi-Expected/hi9String.txt"
outPath = "testing/Hindi-ML/hi9.txt"
scoreName= "hi9 String score"
comments = ""
Stringify = True
print(getScore(refPath,outPath,scoreName,Stringify,comments))

