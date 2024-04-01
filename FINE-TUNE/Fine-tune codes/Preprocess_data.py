from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_metric
metric = load_metric("sacrebleu")

model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"

# Define file paths
en_file_path = 'FINE-TUNE/Fine-tune data/train.en'
hi_file_path = 'FINE-TUNE/Fine-tune data/train.hi'

# Read lines from train.en and train.hi files
with open(en_file_path, 'r', encoding='utf-8') as en_file, open(hi_file_path, 'r', encoding='utf-8') as hi_file:
    en_lines = en_file.readlines()
    hi_lines = hi_file.readlines()

# Create train_data dictionary
train_data = [{"en": en.strip(), "hi": hi.strip()} for en, hi in zip(en_lines, hi_lines)]

# Create Dataset object
train_dataset = Dataset.from_dict({"translation": train_data})

# Create DatasetDict
custom_dataset = DatasetDict({"train": train_dataset})

# Print the structure of custom dataset
print(custom_dataset)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# with tokenizer.as_target_tokenizer():
#     print(tokenizer(["Hello, this one sentence!", "This is another sentence."]))

raw_datasets = custom_dataset

prefix = ""
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "hi"
def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

preprocess_function(raw_datasets['train'][:2])
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)




model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True    
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


import numpy as np
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result



trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()