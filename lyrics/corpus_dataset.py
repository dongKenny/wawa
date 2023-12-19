from datasets import Features, Value, Dataset


def clean_lyrics(lyrics_only=False):
    # False to have cleaned lyrics with the song author/title, True to have only the lyrics for training the RNN
    with open('./lyrics/lyrics.txt', 'r', newline='', encoding='UTF-8') as reader, \
            open('./lyrics/lyrics_cleaned.txt' if not lyrics_only else './lyrics/lyrics_cleaned_rnn.txt', 'w', newline='', encoding='UTF-8') as writer:
        for line in reader.readlines():
            if len(line.split('SEP')) < 3:
                continue
            if lyrics_only:
                writer.write(line.split('SEP')[2])
            else:
                writer.write(line)


def uta_data_generator():
    with open('/raid/kdong4/wawa/lyrics/lyrics.txt', 'r', newline='', encoding='UTF-8') as reader:
        for idx, line in enumerate(reader.readlines()):
            line_data = line.split('SEP')
            if len(line_data) == 3:
                yield {'artist': line_data[0], 'title': line_data[1], 'lyrics': line_data[2]}


def create_uta_dataset():
    dataset = Dataset.from_generator(uta_data_generator, features=Features({'artist': Value(dtype='string'),
                                                                         'title': Value(dtype='string'),
                                                                         'lyrics': Value(dtype='string')}))
    dataset.save_to_disk('uta_dataset.hf')


def tokenize_uta_dataset(song, tokenizer):
    return tokenizer(song['lyrics'], truncation=True, padding=True)


def main():
    clean_lyrics(True)
    # create_uta_dataset()


if __name__ == '__main__':
    main()
