from functions.transcribe import transcribe_audio
from functions.translator import traslatorModel
from functions.lineBreak import getLines
from functions.saveasFile import save
from functions.postProcessHindi import postpHindi
from functions.fbtranslator import fbtranslate
# from functions.ScareBleu import getScore
import os

import time
start_time = time.time()


lines = []
dir = sorted(os.listdir("audiotamilTemp"), key=lambda x: int(x[2:-4]))
print(len(dir))
for audioFile in dir:
    print("processing " +  audioFile)
    file_path = os.path.join("audiotamilTemp", audioFile)
    transcribedText = transcribe_audio(file_path)
    fileLines = getLines(transcribedText, '.')
    # lines.extend("\nFilename = " + audioFile + '\n')
    lines.extend(fileLines)

# Tamil Facebook
tamil = fbtranslate(lines)
save(tamil,"result/tamil","tamil-test_222", asString=0)


# save(lines,"result/english-test","hindi-test",asString=0)
# Hindi
# hindi = postpHindi(traslatorModel(lines,'hi'))
# save(hindi,"result/hindi","devData",asString=1)



# file_path = "audio/ta10.wav"
# transcribedText = transcribe_audio(file_path)
# lines = getLines(transcribedText,'.')
# print(lines)
# save(lines,"result/english-test","tamilTestFile",asString=0)
# # Hindi
# hindi = postpHindi(traslatorModel(lines,'hi'))
# # save(hindi,"result/hindi","forReport",asString=0)

# # Tamil
# tamil = traslatorModel(lines,'dra')
# save(tamil,"result/tamil","forReport", asString=0)

# Tamil Facebook
# tamil = fbtranslate(lines)
# print(tamil)



def print_timer(seconds):
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    print(f"Finish time: {hours:02} : {minutes:02} : {secs:02}")
print_timer(time.time() - start_time)