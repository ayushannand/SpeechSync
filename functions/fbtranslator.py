from transformers import pipeline

def fbtranslate(lines):
    # Initialize the translation pipeline with the specified model
    pipe = pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt")

    result = []
    for line in lines:
        if line.strip():
            # Translate the input text from English (en_XX) to Tamil (ta_IN)
            translated_text = pipe(line, src_lang="en_XX", tgt_lang="ta_IN")
            result.append(translated_text[0]['translation_text'])

    # Print the translated text
    return result

