import string
from random import randint
import ja_core_news_sm
from sklearn.feature_extraction.text import TfidfVectorizer


nlp = ja_core_news_sm.load()
vectorizer = TfidfVectorizer()


def extract_keywords(song, num_keywords=1):
    # Uses spacy to do keyword extraction on Japanese song lyrics and returns num_keyword best keywords/phrases
    doc = nlp(song)
    stop_words = set([word for word in doc if word.is_stop])
    words = [word.text for word in doc.noun_chunks if word not in stop_words]
    tfidf = vectorizer.fit_transform(words)
    keywords = sorted(vectorizer.vocabulary_, key=lambda x: tfidf[0, vectorizer.vocabulary_[x]], reverse=True)[:num_keywords]
    return '、'.join(keywords)


def main():
    with open('../llms/prompt_songs.txt', 'r', encoding='UTF-8') as reader:
        for line in reader.readlines():
            print(extract_keywords(line.split('SEP')[-1]))



if __name__ == '__main__':
    main()
