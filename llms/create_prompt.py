from lyrics.prompt_gen import extract_keywords
import random


def create_lyrics_prompt(num_songs, write_prompt=False):
    # Used to create a prompt with songs from いきものがかり to test the open_calm_7b model
    prompt = 'あなたは日本の作詞家です。あなたに歌の名前と歌詞の例をあげて、いきものがかりのスタイルで歌詞を作成して下さい。そのプロンプト形式は：「曲名」という歌を作って下さい。「曲名」の歌詞は「歌詞」です。次のラインはプロンプトの例です。\n'
    with open('./llms/prompt_songs.txt', 'r') as reader:
        for _ in range(num_songs):
            line = reader.readline().split('SEP')
            title = line[1]
            lyrics = line[2]

            example = f'「{title}」という歌を作って下さい。「{title}」の歌詞は「{lyrics.strip()}」です。'
            prompt = f'{prompt}{example}\n'

    if write_prompt:
        with open('./llms/prompt.txt', 'w') as writer:
            writer.writelines(prompt)
    
    return prompt


def create_speaker_text_prompt(speaker, text):
    return {
        'speaker':speaker,
        'text':text
    }


def create_rinna_prompt(num_songs, artist, song_title, song_keyword, song_data=None, write_prompt=False):
    # Used to create the User/System back-and-forth conversation prompt for the Rinna Instruct model
    prompt = []

    user_context = f'システムは日本の作詞家です。システムに歌の名前と歌詞の例をあげて、{artist}のスタイルで歌詞を作成して下さい。そのプロンプト形式は：「曲名」という歌を作って下さい。{artist}のスタイルで「話題の言葉」の言葉を使って、曲を作成して下さい。'

    songs = song_data[artist]
    for idx, (title, lyrics) in enumerate(random.sample(list(songs.items()), num_songs)):
        user = f'{user_context if idx == 0 else ""}「{title}」という歌を作って下さい。{artist}のスタイルで「{extract_keywords(lyrics)}」の言葉を使って、曲を作成して下さい。'
        system = f'「{title}」の歌詞は「{lyrics.strip()}」です。'
        prompt.append(create_speaker_text_prompt('ユーザー', user))
        prompt.append(create_speaker_text_prompt('システム', system))

    title = song_title
    keyword = song_keyword
    prompt.append(create_speaker_text_prompt('ユーザー', f'「{title}」という歌を作って下さい。{artist}のスタイルで「{keyword}」の言葉を使って、曲を作成して下さい。'))
    prompt = [
        f"{uttr['speaker']}: {uttr['text']}"
        for uttr in prompt
    ]
    prompt = "<NL>".join(prompt)
    prompt = (
        prompt
        + "<NL>"
        + f"システム: 「{title}」の歌詞は「"
    )

    if write_prompt:
        with open('./llms/rinna_prompt.txt', 'w') as writer:
            writer.writelines(prompt.replace("<NL>", "\n"))

    return prompt


def main():
    prompt = create_rinna_prompt(1, True)

    print(prompt)


if __name__ == '__main__':
    main()
