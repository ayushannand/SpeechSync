import whisper

# import ssl

# # Disable SSL certificate verification (use with caution)
# ssl._create_default_https_context = ssl._create_unverified_context

model = whisper.load_model("base")
result = model.transcribe("audio/hi9.wav", fp16=False)
print(result["text"])




