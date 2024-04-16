from functions.transcribe_retarded import transcribe_audio
from functions.translator import traslatorModel
from functions.lineBreak import getLines
from functions.saveasFile import save
from functions.postProcessHindi import postpHindi
from functions.fbtranslator import fbtranslate
from functions.transcriber import transcriber
from functions.ScareBleu import getScore
import os, time


start_time = time.time()

# English transcription, you can remove the yaml parameter if you need to transcribe without it (Forcing line changes based on yaml)
audioFolder = "audioTamil"
yamlFile = "audio/tst-COMMON.yaml"
lines = transcriber(audioFolder, yamlFile)
save(lines,"result/tamil-test","YAMLTranscript",asString=0)


# Tamil Facebook
tamil = fbtranslate(lines)
save(tamil,"result/tamil-test","YAMLTranslate", asString=0)


# save(lines,"result/english-test","hindi-test",asString=0)
# Hindi
# hindi = postpHindi(traslatorModel(lines,'hi'))
# save(hindi,"result/hindi","devData",asString=1)



def print_timer(seconds):
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    print(f"Finish time: {hours:02} : {minutes:02} : {secs:02}")
print_timer(time.time() - start_time)