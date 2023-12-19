def map_artists_to_songs_and_lyrics():
    data = {}

    with open('./lyrics/lyrics_cleaned.txt', 'r', encoding='UTF-8') as reader:
        for line in reader.readlines():
            
            artist, title, lyrics = line.split(' SEP ')
            if artist not in data:
                data[artist] = {}
            data[artist][title] = lyrics

    return data


def create_artist_datalists(data):
    # Creates the HTML datalist options for the deployed rinna page
    result = []
    for artist in data:
        result.append(f'<option value="{artist}">')
    return result


def main():
    data = map_artists_to_songs_and_lyrics()
    print(data['いきものがかり']['アイデンティティ'])
    print(create_artist_datalists(data)[0])


if __name__ == '__main__':
    main()
