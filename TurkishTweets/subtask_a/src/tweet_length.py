from subtask_a import detweetify


def lengths(file_location):
    tweet_lengths = []
    file = open(file_location, 'r')
    lines = file.readlines()

    for line in lines[1:]:
        tweet_lengths.append(len(detweetify(line.split('\t')[1])))

    return tweet_lengths


lens = lengths('../lib/training.tsv')
print(f"Mean: {sum(lens) / len(lens)}")

