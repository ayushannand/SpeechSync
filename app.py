from functions.transcribe_retarded import transcribe_audio
from functions.translator import traslatorModel
from functions.lineBreak import getLines
from functions.saveasFile import save
from functions.postProcessHindi import postpHindi
from functions.ScareBleu import getScore


#Get and save score in 
refPath = "submission/tamilFinal.ta"
outPath = "submission/en-ta_Case_Indic2024.ta"
scoreName= "ta10, fbtranslate, = "
comments = ""
Stringify = True
print(getScore(refPath,outPath,scoreName,Stringify,comments))

