from functions.transcribe_retarded import transcribe_audio
from functions.translator import traslatorModel
from functions.lineBreak import getLines
from functions.saveasFile import save
from functions.postProcessHindi import postpHindi
from functions.ScareBleu import getScore
from functions.swapLines import swapLines


swapLines("result/tamil-test/tamil-testv2-swap.txt","result/tamil-test/tamil-test-googleref.txt")

# Get and save score in 
refPath = "result/tamil-test/tamil-test-googleref.txt"
outPath = "result/tamil-test/tamil-testv2-swap.txt"
scoreName= "ta10, fbtranslate, = "
comments = ""
Stringify = False
print(getScore(refPath,outPath,scoreName,Stringify,comments))

