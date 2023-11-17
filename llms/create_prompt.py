from lyrics.prompt_gen import extract_keywords
import random


def create_lyrics_prompt(num_songs, write_prompt=False):
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


def create_rinna_prompt(num_songs, write_prompt=False):
    prompt = []

    user_context = 'システムは日本の作詞家です。システムに歌の名前と歌詞の例をあげて、いきものがかりのスタイルで歌詞を作成して下さい。そのプロンプト形式は：「曲名」という歌を作って下さい。いきものがかりのスタイルで「話題の言葉」の言葉を使って、曲を作成して下さい。'

    with open('./llms/prompt_songs.txt', 'r') as reader:
        lines = reader.readlines()
        for _ in range(num_songs):
            artist, title, lyrics = random.choice(lines).split('SEP')

            user = f'{user_context}「{title.strip()}」という歌を作って下さい。{artist.strip()}のスタイルで「{extract_keywords(lyrics)}」の言葉を使って、曲を作成して下さい。'
            system = f'「{title}」の歌詞は「{lyrics.strip()}」です。'
            prompt.append(create_speaker_text_prompt('ユーザー', user))
            prompt.append(create_speaker_text_prompt('システム', system))
    
    title = '夏の雨'
    keyword = '雨'
    prompt.append(create_speaker_text_prompt('ユーザー', f'「{title}」という歌を作って下さい。いきものがかりのスタイルで「{keyword}」の言葉を使って、曲を作成して下さい。'))
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
