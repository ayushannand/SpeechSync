# import torch 
# import logging
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# logging.getLogger("transformers").setLevel(logging.ERROR)

# def transcribe_audio(file_path):
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#     # model_id = "openai/whisper-large-v3"
#     # Large v3 is precise but takes a looooot of time
#     model_id = "openai/whisper-tiny"

#     model = AutoModelForSpeechSeq2Seq.from_pretrained(
#         model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
#     )
#     model.to(device)

#     processor = AutoProcessor.from_pretrained(model_id)

#     pipe = pipeline(
#         "automatic-speech-recognition",
#         model=model,
#         tokenizer=processor.tokenizer,
#         feature_extractor=processor.feature_extractor,
#         max_new_tokens=128,
#         chunk_length_s=30,
#         batch_size=16,
#         return_timestamps=True,
#         torch_dtype=torch_dtype,
#         device=device,
#     )

#     result = pipe(file_path)
#     # print(result["text"])
#     return result["text"]


# Use a pipeline as a high-level helper
import logging
from transformers import pipeline
logging.getLogger("transformers").setLevel(logging.ERROR)

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

def transcribe_audio(filePath):
    with open(filePath, "rb") as f:
        # Transcribe audio
        result = pipe(f.read())
    return result["text"]
