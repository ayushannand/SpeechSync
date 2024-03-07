from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

inp = ""
input_ids = tokenizer(inp, return_tensors="pt").input_ids
outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))




# pip install sacremoses
