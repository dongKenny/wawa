from datasets import IterableDataset, Features, Value, Dataset


def uta_data_generator():
    with open('../lyrics/lyrics_cleaned.txt', 'r', newline='', encoding='UTF-8') as reader:
        for idx, line in enumerate(reader.readlines()):
            line_data = line.split('SEP')
            yield {'artist': line_data[0], 'title': line_data[1], 'lyrics': line_data[2]}


def create_uta_dataset():
    return Dataset.from_generator(uta_data_generator, features=Features({'artist': Value(dtype='string'),
                                                                         'title': Value(dtype='string'),
                                                                         'lyrics': Value(dtype='string')}))


def tokenize_uta_dataset(song, tokenizer):
    return tokenizer(song['lyrics'], truncation=True)
