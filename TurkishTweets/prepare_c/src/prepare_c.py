import sys
from sklearn.model_selection import train_test_split


def parse(file_location: str):
    tweets = []
    file = open(file_location, 'r')
    lines = file.readlines()

    id_builder = ""
    line_builder = ""
    multiline = False
    for line in lines[1:]:
        split_line = line.split('\t')
        if len(split_line) == 3 and not multiline:
            multiline = True
            id_builder = split_line[0]
            line_builder = split_line[2]
            continue
        if multiline:
            line_builder += split_line[0]
            if len(split_line) == 2:
                line_builder = line_builder.replace("\n", " ")
                line_builder = line_builder[1:-1]
                tweets.append([id_builder, line_builder, split_line[1].strip()])
                multiline = False
            continue
        tweets.append([split_line[0], split_line[2].strip(), split_line[3].strip()])

    return tweets


if __name__ == "__main__":
    parsed = parse("../lib/troff-v1.0.tsv")
    train, test = train_test_split(parsed, test_size=0.1)
    file = open("../lib/training.tsv", "a")
    file.write("id\ttweet\tsubtask_c\n")
    for line in train:
        file.write(line[0] + "\t" + line[1] + "\t" + line[2] + "\n")

    file1 = open("../lib/test.tsv", "a")
    file2 = open("../lib/labelc.tsv", "a")

    file1.write("id\ttweet\n")
    file2.write("id\tsubtask_b\n")

    for line in test:
        file1.write(line[0] + "\t" + line[1] + "\n")
        file2.write(line[0] + "\t" + line[2] + "\n")

    sys.exit()
