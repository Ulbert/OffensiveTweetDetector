import sys

import json
import emoji
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import Dataset, load_dataset, Value, ClassLabel
import pandas as pd
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def detweetify(text):
    pre_processed = ''
    for token in text.split():
        if token[0] == '@':
            continue
        elif '#' in token:  # TODO: hashtag separation
            continue
        elif any(tld in token for tld in ['.org', '.net', '.com', '.gov', '.edu']):
            continue
        else:
            pre_processed += emoji.demojize(token) + " "

    return pre_processed[:-1]


# normalization is now done through Java, as the Python version of the Zemberek library is too slow and memory-intensive
# processed_test and processed_training are the outputs of the Java program's normalization
#
# extractor = zemberek.TurkishSentenceExtractor()
# normalizer = zemberek.TurkishSentenceNormalizer(zemberek.TurkishMorphology.create_with_defaults())
#
#
# def normalize(text):
#     normalized = ''
#
#     for sentence in extractor.from_paragraph(text):
#         normalized += normalizer.normalize(detweetify(sentence))
#
#     return normalized


device = torch.device('cuda')
batch_size = 64
test_size = 0.1
lr = 1e-5
max_length = 128
epochs = 10


def parse_training(file_location: str):
    tweets_training = []
    file = open(file_location, 'r')
    lines = file.readlines()

    for line in lines[1:]:
        split_line = line.split('\t')

        if split_line[2].strip() == 'oth':
            tweets_training.append([split_line[0], detweetify(split_line[1]), 4])
            tweets_training.append([split_line[0], detweetify(split_line[1]), 4])
            tweets_training.append([split_line[0], detweetify(split_line[1]), 4])
            tweets_training.append([split_line[0], detweetify(split_line[1]), 4])
        elif split_line[2].strip() == 'grp':
            tweets_training.append([split_line[0], detweetify(split_line[1]), 3])
            tweets_training.append([split_line[0], detweetify(split_line[1]), 3])
        elif split_line[2].strip() == 'ind':
            tweets_training.append([split_line[0], detweetify(split_line[1]), 2])
            tweets_training.append([split_line[0], detweetify(split_line[1]), 2])
        elif split_line[2].strip() == 'prof':
            tweets_training.append([split_line[0], detweetify(split_line[1]), 1])
            tweets_training.append([split_line[0], detweetify(split_line[1]), 1])
        else:
            tweets_training.append([split_line[0], detweetify(split_line[1]), 0])

    return Dataset.from_pandas(pd.DataFrame(tweets_training, columns=['id', 'tweet', 'labels']))


# parses both test key and answer key and combines them for easier evaluation
def parse_test_key(test: str, key: str):
    tweets = []
    file1 = open(test, 'r')
    file2 = open(key, 'r')

    file1.readline()
    file2.readline()
    while True:
        line1 = file1.readline()
        line2 = file2.readline()

        if not line1 or not line2:
            break
        split_line1 = line1.split('\t')
        split_line2 = line2.split('\t')

        if split_line2[1].strip() == 'oth':
            tweets.append([split_line1[0], detweetify(split_line1[1]), 4])
        elif split_line2[1].strip() == 'grp':
            tweets.append([split_line1[0], detweetify(split_line1[1]), 3])
        elif split_line2[1].strip() == 'ind':
            tweets.append([split_line1[0], detweetify(split_line1[1]), 2])
        elif split_line2[1].strip() == 'prof':
            tweets.append([split_line1[0], detweetify(split_line1[1]), 1])
        else:
            tweets.append([split_line1[0], detweetify(split_line1[1]), 0])

    return Dataset.from_pandas(pd.DataFrame(tweets, columns=['id', 'tweet', 'labels']))


tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-128k-uncased')
model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-128k-uncased', num_labels=5, gradient_checkpointing=True)


def tokenize(row):
    return tokenizer(row['tweet'], padding='max_length', truncation=True, max_length=max_length)


def train(training: Dataset):
    print("Splitting files into training and dev...")
    dataset = training.train_test_split(test_size=test_size)

    print("Tokenizing...")
    dataset['train'] = dataset['train'].map(tokenize, batched=True)
    dataset['train'].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    dataloader = DataLoader(dataset['train'], batch_size=batch_size)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("|               Training                |")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    optimizer = AdamW(params=model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train().to(device)
        print(f"\n\nEpoch {epoch + 1}...")
        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, y_pred = model(**batch)
            loss.backward()
            optimizer.step()
            model.zero_grad()

    return model


def evaluate(dataset: Dataset):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("|              Evaluation               |")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Tokenizing...")
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model.eval().to(device)

    print("Evaluating...")
    y_preds = []
    for i, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch)[1]
        y_preds += list(np.argmax(logits.detach().cpu().numpy(), axis=1).flatten())
        model.zero_grad()

    print(classification_report(dataset['labels'], y_preds))
    return f1_score(dataset['labels'], y_preds, average='macro')


if __name__ == "__main__":
    train(parse_training('../lib/training.tsv'))
    torch.save(model.state_dict(), "../model.pt")

    # model.load_state_dict(torch.load("../model.pt"))
    evaluate(parse_test_key("../lib/test.tsv", "../lib/labelc.tsv"))

    sys.exit()
