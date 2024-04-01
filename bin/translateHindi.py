from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def getHindi(lines): 
    # Load translation model
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

    # Translate each line using the Helsinki model
    result = []
    for line in lines:
        # Translate the line using the model
        inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True)
        translated_line = model.generate(**inputs)
        
        # Decode the translated output
        translated_text = tokenizer.decode(translated_line[0], skip_special_tokens=True)
        
        # Append the translated text to the result array
        result.append(translated_text)

    return result
