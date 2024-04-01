from functions.transcribe import transcribe_audio
from functions.translator import traslatorModel
from functions.lineBreak import getLines
from functions.saveasFile import save


file_path = "audio/hi9.wav"
transcribedText = transcribe_audio(file_path)
lines = getLines(transcribedText,'.')

# Hindi
hindi = traslatorModel(lines,'hi')
save(hindi,"result/hindi","hi9")

# Tamil
tamil = traslatorModel(lines,'dra')
save(tamil,"result/tamil","op")
