import string
from random import randint
import ja_core_news_sm
from sklearn.feature_extraction.text import TfidfVectorizer


nlp = ja_core_news_sm.load()
vectorizer = TfidfVectorizer()


def extract_keywords(song, num_keywords=1):
    doc = nlp(song)
    stop_words = set([word for word in doc if word.is_stop])
    words = [word.text for word in doc.noun_chunks if word not in stop_words]
    tfidf = vectorizer.fit_transform(words)
    keywords = sorted(vectorizer.vocabulary_, key=lambda x: tfidf[0, vectorizer.vocabulary_[x]], reverse=True)[:num_keywords]
    return '、'.join(keywords)


def main():
    # with open('../llms/prompt_songs.txt', 'r', encoding='UTF-8') as reader:
    #     for line in reader.readlines():
    #         print(extract_keywords(line.split('SEP')[-1]))
    
    print(extract_keywords("""突然降る夕立 あぁ傘もないや嫌
空のご機嫌なんか知らない
季節の変わり目の服は 何着りゃいいんだろ
春と秋 どこいっちゃったんだよ
息も出来ない 情報の圧力
めまいの螺旋だ わたしはどこにいる
こんなに こんなに 息の音がするのに
変だね 世界の音がしない
足りない 足りない 誰にも気づかれない
殴り書きみたいな音 出せない状態で叫んだよ
「ありのまま」なんて 誰に見せるんだ
馬鹿なわたしは歌うだけ
ぶちまけちゃおうか 星に
エリクサーに張り替える作業もなんとなくなんだ
欠けた爪を少し触る
半径300mmの体で 必死に鳴いてる
音楽にとっちゃ ココが地球だな
空気を握って 空を殴るよ
なんにも起きない わたしは無力さ
だけどさ その手で この鉄を弾いたら
何かが変わって見えた ような
眩しい 眩しい そんなに光るなよ
わたしのダサい影が より色濃くなってしまうだろ
なんでこんな熱くなっちゃってんだ 止まんない
馬鹿なわたしは歌うだけ
うるさいんだって 心臓
蒼い惑星 ひとりぼっち
いっぱいの音を聞いてきた
回り続けて 幾億年
一瞬でもいいから ああ
聞いて
聴けよ
わたし わたし わたしはここにいる
殴り書きみたいな音 出せない状態で叫んだよ
なんかになりたい なりたい 何者かでいい
馬鹿なわたしは歌うだけ
ぶちまけちゃおうか 星に"""))


if __name__ == '__main__':
    main()
