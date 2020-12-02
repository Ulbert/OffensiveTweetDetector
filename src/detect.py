import sys

from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import Dataset, load_dataset, Value, ClassLabel
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score

# normalization is now done through Java, as the Python version of the Zemberek library is too slow and memory-intensive
# processed_test and processed_training are the outputs of the Java program's normalization

# def detweetify(text):
#     pre_processed = ''
#     for token in text.split():
#         if token[0] == '@':
#             continue
#         elif '#' in token:  # TODO: hashtag separation
#             continue
#         elif any(tld in token for tld in ['.org', '.net', '.com', '.gov', '.edu']):
#             continue
#         else:
#             pre_processed += emoji.demojize(token) + " "
#
#     return pre_processed[:-1]
#
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
max_length = 64
epochs = 3


def parse(file_location: str):
    tweets = []
    file = open(file_location, 'r')
    lines = file.readlines()

    if len(lines[0].split()) == 2:  # answer key
        for line in lines[1:]:
            split_line = line.split('\t')
            tweets.append([split_line[0], split_line[1] == 'OFF'])
    else:  # test/training key
        for line in lines[1:]:
            split_line = line.split('\t')
            if split_line[2].strip() == 'OFF':
                tweets.append([split_line[0], split_line[1], 1])
            else:
                tweets.append([split_line[0], split_line[1], 0])

    return Dataset.from_pandas(pd.DataFrame(tweets, columns=['id', 'tweet', 'labels']))


tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-128k-uncased')
model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-128k-uncased')


def tokenize(row):
    return tokenizer(row['tweet'], padding='max_length', truncation=True, max_length=64)


def train(training: Dataset):
    print("Splitting files into training and dev...")
    dataset = training.train_test_split(test_size=test_size)

    print("Tokenizing...")
    dataset['train'] = dataset['train'].map(tokenize, batched=True)
    dataset['test'] = dataset['test'].map(tokenize, batched=True)

    print("Constructing DataLoader...")
    dataset['train'].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    dataset['test'].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    dataloader = dict()
    dataloader['train'] = DataLoader(dataset['train'], batch_size=batch_size)
    dataloader['test'] = DataLoader(dataset['test'], batch_size=batch_size)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("|            Begin Training             |")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    optimizer = AdamW(params=model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train().to(device)
        print(f"\n\nEpoch {epoch + 1}...")
        for i, batch in enumerate(tqdm(dataloader['train'])):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, y_pred = model(**batch)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            # optimizer.zero_grad()

            # if i % 50:
            #     print(f"loss: {loss}")

        print("Evaluating...")
        evaluate(dataloader['test'])
    return model


def evaluate(dataloader: DataLoader):
    model.eval().to(device)

    test_acc = 0.0
    test_f1 = 0.0
    for i, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        y_pred = model(**batch)[1]
        y_pred = torch.argmax(y_pred, dim=-1)
        model.zero_grad()

        test_acc += accuracy_score(batch['labels'].cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        test_f1 += f1_score(batch['labels'].cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='weighted')
    print(f"Accuracy: {test_acc} F1-Score: {test_f1}")


if __name__ == "__main__":
    training_set = parse('../lib/processed_training.tsv')
    torch.save(train(training_set).state_dict(), "../model.pt")
    sys.exit()
