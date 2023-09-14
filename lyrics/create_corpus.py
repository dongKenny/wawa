def strip_author_and_title():
    with open('lyrics.txt', 'r', newline='', encoding='UTF-8') as reader,\
            open('lyrics_corpus.txt', 'w', newline='', encoding='UTF-8') as writer:
        for line in reader.readlines():
            writer.write(f'{line.split("SEP")[-1]}')


def main():
    strip_author_and_title()


if __name__ == '__main__':
    main()
