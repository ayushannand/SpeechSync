#!/usr/bin/env python
# coding: utf-8

# In[16]:


import warnings
import numpy as np
import pandas as pd

import torch
import transformers

from datasets import Dataset
from datasets import load_metric

from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

warnings.filterwarnings("ignore")


# In[213]:


BATCH_SIZE = 16
BLEU = "bleu"
HINDI = "hi"
HINDI_TEXT = "hindi_text"
EPOCH = "epoch"
INPUT_IDS = "input_ids"
FILENAME = "FINE-TUNE/Fine-tune data/output.csv"
GEN_LEN = "gen_len"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-en-hi"
MODEL_NAME = MODEL_CHECKPOINT.split("/")[-1]
LABELS = "labels"
PREFIX = ""
ENGLISH = "en"
ENGLISH_TEXT = "english_text"
SCORE = "score"
SOURCE_LANG = "en"
TARGET_LANG = "hi"
TRANSLATION = "translation"
UNNAMED_COL = "Unnamed: 0"
     


# In[214]:


def postprocess_text(preds: list, labels: list) -> tuple:
    """Performs post processing on the prediction text and labels"""

    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def prep_data_for_model_fine_tuning(source_lang: list, target_lang: list) -> list:
    """Takes the input data lists and converts into translation list of dicts"""

    data_dict = dict()
    data_dict[TRANSLATION] = []

    for sr_text, tr_text in zip(source_lang, target_lang):
        temp_dict = dict()
        temp_dict[ENGLISH] = sr_text
        temp_dict[HINDI] = tr_text

        data_dict[TRANSLATION].append(temp_dict)

    return data_dict


def generate_model_ready_dataset(dataset: list, source: str, target: str,
                                 model_checkpoint: str,
                                 tokenizer: AutoTokenizer):
    """Makes the data training ready for the model"""

    preped_data = []

    for row in dataset:
        inputs = PREFIX + row[source]
        targets = row[target]

        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH,
                                 truncation=True, padding=True)

        model_inputs[TRANSLATION] = row

        # setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=MAX_INPUT_LENGTH,
                                 truncation=True, padding=True)
            model_inputs[LABELS] = labels[INPUT_IDS]

        preped_data.append(model_inputs)

    return preped_data



def compute_metrics(eval_preds: tuple) -> dict:
    """computes bleu score and other performance metrics """

    metric = load_metric("sacrebleu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

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
    result = {BLEU: result[SCORE]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

    result[GEN_LEN] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result


# In[215]:

# Example usage:
file1 = 'FINE-TUNE/Fine-tune data/dev.en'
file2 = 'FINE-TUNE/Fine-tune data/dev.hi'

def merge_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

        # Check if both files have the same number of lines
        if len(lines1) != len(lines2):
            print("Error: Both files don't have the same number of lines.")
            return None

        # Create a DataFrame with two columns, one for each file
        data = {'english_text': [line.strip() for line in lines1],
                'hindi_text': [line.strip() for line in lines2]}

        return pd.DataFrame(data)



translation_data = merge_files(file1, file2)
translation_data


# In[216]:


X = translation_data[ENGLISH_TEXT]
y = translation_data[HINDI_TEXT]


# In[217]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10,
                                                    shuffle=True,
                                                    random_state=100)

print("INITIAL X-TRAIN SHAPE: ", x_train.shape)
print("INITIAL Y-TRAIN SHAPE: ", y_train.shape)
print("X-TEST SHAPE: ", x_test.shape)
print("Y-TEST SHAPE: ", y_test.shape)


# In[218]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.20,
                                                  shuffle=True,
                                                  random_state=100)

print("FINAL X-TRAIN SHAPE: ", x_train.shape)
print("FINAL Y-TRAIN SHAPE: ", y_train.shape)
print("X-VAL SHAPE: ", x_val.shape)
print("Y-VAL SHAPE: ", y_val.shape)


# In[219]:


tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)


# In[220]:


training_data = prep_data_for_model_fine_tuning(x_train.values, y_train.values)

validation_data = prep_data_for_model_fine_tuning(x_val.values, y_val.values)

test_data = prep_data_for_model_fine_tuning(x_test.values, y_test.values)
     


# In[221]:


train_data = generate_model_ready_dataset(dataset=training_data[TRANSLATION],
                                          tokenizer=tokenizer,
                                          source=ENGLISH,
                                          target=HINDI,
                                          model_checkpoint=MODEL_CHECKPOINT)

validation_data = generate_model_ready_dataset(dataset=validation_data[TRANSLATION],
                                               tokenizer=tokenizer,
                                               source=ENGLISH,
                                               target=HINDI,
                                               model_checkpoint=MODEL_CHECKPOINT)

test_data = generate_model_ready_dataset(dataset=test_data[TRANSLATION],
                                               tokenizer=tokenizer,
                                               source=ENGLISH,
                                               target=HINDI,
                                               model_checkpoint=MODEL_CHECKPOINT)


# In[222]:


train_df = pd.DataFrame.from_records(train_data)
train_df.info()


# In[223]:


validation_df = pd.DataFrame.from_records(validation_data)
validation_df.info()
     




# In[224]:


test_df = pd.DataFrame.from_records(test_data)
test_df.info()


# In[225]:


train_dataset = Dataset.from_pandas(train_df)
train_dataset


# In[226]:


validation_dataset = Dataset.from_pandas(validation_df)
validation_dataset
     


# In[227]:


test_dataset = Dataset.from_pandas(test_df)
test_dataset


# In[228]:


model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)


# In[229]:


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# In[230]:


model_args = Seq2SeqTrainingArguments(
    f"{MODEL_NAME}-finetuned-{SOURCE_LANG}-to-{TARGET_LANG}",
    evaluation_strategy=EPOCH,
    learning_rate=2e-4,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.02,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True
)


# In[231]:


trainer = Seq2SeqTrainer(
    model,
    model_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
    


# In[191]: Train the model

trainer.train()


# In[ ]: save fine tuned model
trainer.save_model("FineTunedTransformer")

# predict test data
test_results = trainer.predict(test_dataset)

#print BLUEScore
print("Test Bleu Score: ", test_results.metrics["test_bleu"])

