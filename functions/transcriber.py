import os
import yaml
# from functions.transcribe import transcribe_audio
from functions.translator import traslatorModel
from functions.lineBreak import getLines
from functions.saveasFile import save
from functions.postProcessHindi import postpHindi
from functions.fbtranslator import fbtranslate
from pydub import AudioSegment
import logging
from transformers import pipeline
logging.getLogger("transformers").setLevel(logging.ERROR)

#Whisper transcriber
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
def transcribe_audio(filePath):
    with open(filePath, "rb") as f:
        result = pipe(f.read())
    return result["text"]

def transcriber(audios_dir, yaml_file_path):
    # Load the YAML file containing audio segments
    with open(yaml_file_path, "r") as file:
        audio_segments = yaml.safe_load(file)
    
    transcriptions = []
    for audio_file in sorted(os.listdir(audios_dir), key=lambda x: int(x[2:-4])):
        print(f"Processing {audio_file}")
        segments = [seg for seg in audio_segments if seg['wav'] == audio_file]
        audio = AudioSegment.from_file(os.path.join(audios_dir, audio_file))
        for segment in segments:
            offset_ms = segment['offset'] * 1000 
            duration_ms = segment['duration'] * 1000 
            audio_segment = audio[offset_ms:offset_ms + duration_ms]
            temp_audio_file = "bin/temp_audio.wav"
            audio_segment.export(temp_audio_file, format="wav")
            transcribed_text = transcribe_audio(temp_audio_file)
            transcriptions.append(transcribed_text.strip())

    return transcriptions



# def transcriber(audios_dir):
#     lines = []
#     dir = sorted(os.listdir(audios_dir), key=lambda x: int(x[2:-4]))
#     for audioFile in dir:
#         print("processing " +  audioFile)
#         file_path = os.path.join("audiotamilTemp", audioFile)
#         transcribedText = transcribe_audio(file_path)
#         fileLines = getLines(transcribedText, '.')
#         # lines.extend("\nFilename = " + audioFile + '\n')
#         lines.extend(fileLines)
#     return lines