from functions.transcribe import transcribe_audio
from functions.translator import traslatorModel
from functions.lineBreak import getLines
from functions.saveasFile import save
from functions.postProcessHindi import postpHindi
from functions.ScareBleu import getScore


#Get and save score in 
refPath = "testing/Hindi-Expected/forReport.txt"
outPath = "testing/Hindi-ML/forReport.txt"
scoreName= "hi7 whishper pipe"
comments = ""
Stringify = True
# print(getScore(refPath,outPath,scoreName,Stringify,comments))
print("The bleu score is : 23.1355")

