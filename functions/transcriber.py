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

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
def transcribe_audio(filePath):
    with open(filePath, "rb") as f:
        # Transcribe audio
        result = pipe(f.read())
    return result["text"]

def transcriber(audios_dir, yaml_file_path):
    # Load the YAML file containing audio segments
    with open(yaml_file_path, "r") as file:
        audio_segments = yaml.safe_load(file)
    
    # List to store the transcriptions
    transcriptions = []

    # Iterate through audio files
    for audio_file in sorted(os.listdir(audios_dir), key=lambda x: int(x[2:-4])):
        print(f"Processing {audio_file}")

        # Find segments for the current audio file
        segments = [seg for seg in audio_segments if seg['wav'] == audio_file]

        # Load the audio file
        audio = AudioSegment.from_file(os.path.join(audios_dir, audio_file))

        # Process each segment
        for segment in segments:
            offset_ms = segment['offset'] * 1000  # Convert offset to milliseconds
            duration_ms = segment['duration'] * 1000  # Convert duration to milliseconds

            # Extract the segment from the audio file
            audio_segment = audio[offset_ms:offset_ms + duration_ms]

            # Export the audio segment to a temporary file
            temp_audio_file = "temp_audio.wav"
            audio_segment.export(temp_audio_file, format="wav")

            # Transcribe the audio segment
            transcribed_text = transcribe_audio(temp_audio_file)

            # Append the cleaned transcription
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