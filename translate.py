from functions.transcribe import transcribe_audio
from functions.translator import traslatorModel
from functions.lineBreak import getLines
from functions.saveasFile import save
from functions.postProcessHindi import postpHindi
# from functions.ScareBleu import getScore
import os

import time


start_time = time.time()


lines = []
dir = sorted(os.listdir("devAudio"), key=lambda x: int(x[2:-4]))
print(len(dir))
for audioFile in dir:
    print("processing " +  audioFile)
    file_path = os.path.join("devAudio", audioFile)
    transcribedText = transcribe_audio(file_path)
    fileLines = getLines(transcribedText, '.')
    lines.extend(fileLines)

save(lines,"result/english","devData",asString=1)
# Hindi
hindi = postpHindi(traslatorModel(lines,'hi'))
save(hindi,"result/hindi","devData",asString=1)



# file_path = "audio/hi7.wav"
# transcribedText = transcribe_audio(file_path)
# lines = getLines(transcribedText,'.')
# print(lines)
# # Hindi
# hindi = postpHindi(traslatorModel(lines,'hi'))
# save(hindi,"result/hindi","pipe_hi7",asString=1)

# # Tamil
# tamil = traslatorModel(lines,'dra')
# save(tamil,"result/tamil","hi7")


def print_timer(seconds):
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    print(f"Finish time: {hours:02} : {minutes:02} : {secs:02}")
print_timer(time.time() - start_time)