import json
import re
import nltk

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer


def create_data_list(file_path):
    with open(file_path, 'r') as file_json:
        data = json.load(file_json)
    
    all_words = []
    all_labels = []
    for key in data:
        processed_article = data[key]['headline'].lower()
        processed_article = re.sub('[^a-zA-Z]', ' ', processed_article)
        processed_article = re.sub(r'\s+', ' ', processed_article)

        words_list = nltk.word_tokenize(processed_article) 
        all_words.append(words_list)
        all_labels.append(data[key]['is_sarcastic'])

    return all_words, all_labels


def create_word2vec(all_words_list):
    word2vec = Word2Vec(all_words_list, min_count=2, sg=1)
    # vocabulary = word2vec.wv.vocab
    return word2vec


def create_tfidf(all_words_list):
    vectorizer = TfidfVectorizer()
    sentences_list = []
    for sentence in all_words_list:
        ps = PorterStemmer()
        sentence = [ps.stem(word) for word in sentence]
        sentences_list.append(' '.join(sentence))
        # if ' '.join(sentece1) != ' '.join(sentece):
        #     print("With stem: "+ ' '.join(sentece1))
        #     print("No stem:   "+' '.join(sentece))

    tfidf = vectorizer.fit_transform(sentences_list).toarray()
    return tfidf
