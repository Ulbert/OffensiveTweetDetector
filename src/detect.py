import transformers
import numpy as np
import zemberek
import pandas as pd
import torch


def detweetify(text):
    pre_processed = ''
    for token in text.split():
        if token[0] == '@':
            continue
        elif any(tld in token for tld in ['.org', '.net', '.com', '.gov', '.edu']):
            continue
        else:
            pre_processed.join(token).join(' ')

    return pre_processed


def normalize(text):
    normalized = ''

    morphology = zemberek.TurkishMorphology.create_with_defaults()
    extractor = zemberek.TurkishSentenceExtractor()
    normalizer = zemberek.TurkishSentenceNormalizer(morphology)

    for sentence in extractor.from_paragraph(text):
        normalized.join(normalizer.normalize(sentence))

    return normalized


def preprocess(text):
    return normalize(detweetify(text)).lower()


def parse(file_location):
    tweet_tuples = []
    file = open(file_location, 'r')
    lines = file.readlines()

    if len(lines[0].split()) == 2:    # answer key
        for line in lines[1:]:
            split_line = line.split()
            tweet_tuples.append([split_line[0], split_line[1] == 'OFF'])
    else:                             # test/training key
        for line in lines[1:]:
            split_line = line.split()
            tweet_tuples.append([split_line[0], preprocess(split_line[1]), split_line[2] == 'OFF'])

    return tweet_tuples


tokenizer = transformers.AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-128k-uncased').to(torch.device('cuda'))
model = transformers.AutoModel.from_pretrained('dbmdz/bert-base-turkish-128k-uncased').to(torch.device('cuda'))


def tokenize(tuple_list):
    df = pd.DataFrame(tuple_list, columns=['id', 'tweet', 'subtask_a'])

    for row in tuple_list:
        df.append([row[0], tokenizer.encode(row[1]), row[2]])

    return df
