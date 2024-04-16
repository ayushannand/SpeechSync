import logging
from transformers import pipeline
logging.getLogger("transformers").setLevel(logging.ERROR)

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

def transcribe_audio(filePath):
    with open(filePath, "rb") as f:
        # Transcribe audio
        result = pipe(f.read())
    return result["text"]



#Discarded file