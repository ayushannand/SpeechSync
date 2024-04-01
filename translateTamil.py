from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app import transcribe_audio
import re

# Load translation model
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-dra")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-dra")

# Transcribe audio and translate
file_path = "audio/hi7.wav"
transcribed_text = transcribe_audio(file_path)

# Preprocess transcribed text
transcribed_text = re.sub(r'[\n\r"]', '', transcribed_text)  # Remove newline, carriage return, and quotes

# Define maximum sequence length for translation
max_length = 512

# Split transcribed text into segments
segments = [transcribed_text[i:i+max_length] for i in range(0, len(transcribed_text), max_length)]

# Translate each segment
translated_texts = []
for segment in segments:
    input_ids = tokenizer(segment, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
    translated_texts.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

print("Transcribed Text:", transcribed_text)
print("Translated Texts:")
for text in translated_texts:
    print(text)



# pip install sacremoses


# # Usage example:
# file_path = "audio/hi13.wav"
# transcribed_text = transcribe_audio(file_path)
# print(transcribed_text)
