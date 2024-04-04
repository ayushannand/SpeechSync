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


BATCH_SIZE = 16
BLEU = "bleu"
ENGLISH = "en"
ENGLISH_TEXT = "english_text"
EPOCH = "epoch"
INPUT_IDS = "input_ids"
FILENAME = "TranslationDataset.csv"
GEN_LEN = "gen_len"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
MODEL_CHECKPOINT = "unicamp-dl/translation-pt-en-t5"
MODEL_NAME = MODEL_CHECKPOINT.split("/")[-1]
LABELS = "labels"
PREFIX = ""
PORTUGUESE = "pt"
PORTUGUESE_TEXT = "portuguese_text"
SCORE = "score"
SOURCE_LANG = "pt"
TARGET_LANG = "en"
TRANSLATION = "translation"
UNNAMED_COL = "Unnamed: 0"

# Read file 1 (Portuguese text)
with open('FINE-TUNE/Fine-tune data/train.en', 'r', encoding='utf-8') as f1:
    source_lines = f1.readlines()

# Read file 2 (English text)
with open('FINE-TUNE/Fine-tune data/train.en', 'r', encoding='utf-8') as f2:
    target_lines = f2.readlines()

