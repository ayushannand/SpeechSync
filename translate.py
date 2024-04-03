from functions.transcribe import transcribe_audio
from functions.translator import traslatorModel
from functions.lineBreak import getLines
from functions.saveasFile import save
from functions.postProcessHindi import postpHindi
from functions.ScareBleu import getScore

file_path = "audio/hi9.wav"
transcribedText = transcribe_audio(file_path)
lines = getLines(transcribedText,'.')

print(lines)

# Hindi
hindi = postpHindi(traslatorModel(lines,'hi'))
save(hindi,"result/hindi","temp",asString=1)

# # Tamil
# tamil = traslatorModel(lines,'dra')
# save(tamil,"result/tamil","hi7")
