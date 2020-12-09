from src.detect import detweetify


def max_length(file_location):
    record = 0
    file = open(file_location, 'r')
    lines = file.readlines()

    for line in lines[1:]:
        if len(detweetify(line.split('\t')[1])) > record:
            record = len(line)

    return record


print(max_length('../lib/training.tsv'))
