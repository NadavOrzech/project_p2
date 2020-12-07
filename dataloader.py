from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import nltk
from config import Config 
import json
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import sklearn


class Dataloader():
    def __init__(self, params, generate_tfidf=True, feature_flag=True):
        self.input_path = params.input_path
        self.num_features = len(params.features_map)
        self.features_map = params.features_map
        self.interjections = params.interjections
        self.intensifiers = params.intensifiers
        self.features_flag=feature_flag

        self.all_words, self.all_labels = self.create_data_list()
        if generate_tfidf:
            self.tfidf_matrix = self.generate_tfidf_matrix()
            self.x_train, self.x_test, self.y_train, self.y_test, self.tfidf_train, self.tfidf_test = sklearn.model_selection.train_test_split(self.all_words, self.all_labels, self.tfidf_matrix, test_size=0.15, random_state=42)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(self.all_words, self.all_labels, test_size=0.15, random_state=42)


        self.sarcastic_common_words = None
        self.not_sarcastic_common_words = None
        self.get_common_words(self.x_train,self.y_train)
        
        # self.features_matrix = np.zeros((len(self.x_train),self.num_features))
        # self.extract_raw_features(self.x_train)
        # self.extract_initial_process_features(self.x_train)
        # # self.x = np.concatenate((self.features_matrix,self.tfidf_matrix), axis=1)
        aaa =3

    def create_data_list(self):
        with open(self.input_path, 'r') as file_json:
            data = json.load(file_json)
        
        all_words = []
        all_labels = []
        for key in data:
            headline = data[key]['headline'].lower()
            all_words.append(headline)
            all_labels.append(data[key]['is_sarcastic'])

        return all_words, all_labels
    
    def generate_tfidf_matrix(self):
        all_words_list = []
        for i,sentence in enumerate(self.all_words):
            processed_article = re.sub('[^a-zA-Z]', ' ', sentence)
            processed_article = re.sub(r'\s+', ' ', processed_article)
            words_list = nltk.word_tokenize(processed_article) 
            all_words_list.append(words_list)       
        
        tfidf_matrix = create_tfidf(all_words_list)

        return tfidf_matrix

    def get_train_dataloader(self, tfidf=False):
        if tfidf:
            features_matix =  self.generate_features_matrix(self.x_train, self.tfidf_train)
        else:
            features_matix =  self.generate_features_matrix(self.x_train)

        return features_matix, self.y_train

    def get_test_dataloader(self, tfidf=False):
        if tfidf:
            features_matix = self.generate_features_matrix(self.x_test, self.tfidf_test)
        else:
            features_matix = self.generate_features_matrix(self.x_test)

        return features_matix, self.y_test


    def generate_features_matrix(self, sentences_list, tfidf_matrix=None):
        features_matrix = np.zeros((len(sentences_list),self.num_features))
        all_words_list = []

        for i,sentence in enumerate(sentences_list):

            processed_article = re.sub('[^a-zA-Z]', ' ', sentence)
            processed_article = re.sub(r'\s+', ' ', processed_article)
            words_list = nltk.word_tokenize(processed_article) 
            word_tags = nltk.pos_tag(words_list)
            sentence_length = len(word_tags)


            # punctuation marks binary flag  
            punctuations = 0
            if self.features_flag:
                if sentence.count('?') > 0 or sentence.count('!') > 0:
                    punctuations = 1
            else:
                punctuations = (sentence.count('?') + sentence.count('!'))/sentence_length
            
            features_matrix[i,self.features_map['punctuations']] = punctuations

            # quotes marks binary flag
            quotes = 0
            if self.features_flag:
                if sentence.count('\"') > 0 or sentence.count('\'') > 0:
                    quotes = 1
            else:
                quotes = (sentence.count('\"') + sentence.count('\''))/sentence_length
            
            features_matrix[i,self.features_map['quotes']] = quotes

            adjective = 0
            adverb = 0
            interjection = 0
            positive_word = 0
            negative_word = 0
            sarcastic_word = 0
            not_sarcastic_word = 0
            
            if self.features_flag:
                for w, tag in word_tags:
                    # binary flag for adjectives, adverbs and interjections per example
                    if tag in ["JJ", "JJR", "JJS"]:
                        adjective = 1
                    elif tag in ["RR", "RBR", "RBS"]:
                        adverb = 1
                    elif w in self.interjections:
                        interjection = 1

                    # binary flag for positive sentiment or negative sentiment words
                    word_blob = TextBlob(w)
                    if word_blob.sentiment.polarity >= 0.5:
                        positive_word=1
                    elif word_blob.sentiment.polarity <= -0.5:
                        negative_word=1

                    if w in self.sarcastic_common_words:
                        sarcastic_word=1
                    elif w in self.not_sarcastic_common_words:
                        not_sarcastic_word=1
            else:
                for w, tag in word_tags:
                    # binary flag for adjectives, adverbs and interjections per example
                    if tag in ["JJ", "JJR", "JJS"]:
                        adjective += 1
                    elif tag in ["RR", "RBR", "RBS"]:
                        adverb += 1
                    elif w in self.interjections:
                        interjection += 1

                    # binary flag for positive sentiment or negative sentiment words
                    word_blob = TextBlob(w)
                    if word_blob.sentiment.polarity >= 0.5:
                        positive_word+=1
                    elif word_blob.sentiment.polarity <= -0.5:
                        negative_word+=1

                    if w in self.sarcastic_common_words:
                        sarcastic_word+=1
                    elif w in self.not_sarcastic_common_words:
                        not_sarcastic_word+=1
                
                adjective /= sentence_length
                adverb /= sentence_length
                interjection /= sentence_length
                positive_word /= sentence_length
                negative_word /= sentence_length
                sarcastic_word /= sentence_length
                not_sarcastic_word /= sentence_length
                
            # adverbs_count += adverb
            # adjective_count+=adjective
            # interjections_count +=interjection
            features_matrix[i,self.features_map['adjectives']] = adjective
            features_matrix[i,self.features_map['adverbs']] = adverb
            features_matrix[i,self.features_map['interjections']] = interjection
            features_matrix[i,self.features_map['positive_word']] = positive_word
            features_matrix[i,self.features_map['negative_word']] = negative_word
            features_matrix[i,self.features_map['sentence_length']] = sentence_length
            features_matrix[i,self.features_map['common_sarcastic']] = sarcastic_word
            features_matrix[i,self.features_map['common_not_sarcastic']] = not_sarcastic_word
            
            all_words_list.append(words_list)       
        
        if tfidf_matrix is not None:
            features_matrix = np.concatenate((features_matrix,tfidf_matrix), axis=1)

        return features_matrix

    def get_common_words(self, sentences_list, labels_list, proportion=4, k=50):
        words_dic = {}
        for i, sentence in enumerate(sentences_list):
            processed_article = re.sub('[^a-zA-Z]', ' ', sentence)
            processed_article = re.sub(r'\s+', ' ', processed_article)
            words_list = nltk.word_tokenize(processed_article) 
            # TODO: consider stemming?
            for word in words_list:
                if word in words_dic.keys():
                    if labels_list[i]:
                        words_dic[word]['pos']+=1
                    else:
                        words_dic[word]['neg']+=1
                else:
                    if labels_list[i]:
                        words_dic[word] = {
                                'pos': 1,
                                'neg': 0
                            }
                    else:
                        words_dic[word] = {
                                'pos': 0,
                                'neg': 1
                            }
            
        positive_common_instances, negative_common_instances = [],[]        
        for word in words_dic:
            # TODO: consider removing stop words?
            pos = words_dic[word]['pos']
            neg = words_dic[word]['neg']
            if pos == 0 or neg == 0:
                continue
            if pos >= proportion * neg:
                positive_common_instances.append((word,(pos,neg)))
            elif neg >= proportion * pos:
                negative_common_instances.append((word,(neg,pos)))

        positive_common_instances = sorted(positive_common_instances, key=lambda tup: tup[1][0])[-k:]
        negative_common_instances = sorted(negative_common_instances, key=lambda tup: tup[1][0])[-k:]
        
        self.sarcastic_common_words = [w for w,tup in positive_common_instances]
        self.not_sarcastic_common_words = [w for w,tup in negative_common_instances]

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


if __name__ == "__main__":
    params = Config()
    dataloader = Dataloader(params)

    


    # text          = "I feel the product is so good" 
    # # text = "mother comes pretty close to using word 'streaming' correctly"
    # sent          = TextBlob(text)
    # # The polarity score is a float within the range [-1.0, 1.0]
    # # where negative value indicates negative text and positive
    # # value indicates that the given text is positive.
    # polarity      = sent.sentiment.polarity
    # # The subjectivity is a float within the range [0.0, 1.0] where
    # # 0.0 is very objective and 1.0 is very subjective.
    # subjectivity  = sent.sentiment.subjectivity

    # sent          = TextBlob(text, analyzer = NaiveBayesAnalyzer())
    # classification= sent.sentiment.classification
    # positive      = sent.sentiment.p_pos
    # negative      = sent.sentiment.p_neg

    # print(polarity,subjectivity,classification,positive,negative)

'''        
    def extract_raw_features(self,sentences_list):
        for i,sentence in enumerate(sentences_list):
            
            punctuations = 0
            if sentence.count('?') > 0 or sentence.count('!') > 0:
                punctuations = 1
            self.features_matrix[i,self.features_map['punctuations']] = punctuations

            quotes = 0
            if sentence.count('\"') > 0 or sentence.count('\'') > 0:
                quotes = 1
            self.features_matrix[i,self.features_map['quotes']] = quotes

            
            # text =  TextBlob(sentence)
            # self.features_matrix[i,self.features_map['polarity']] = text.sentiment.polarity
            # self.features_matrix[i,self.features_map['subjectivity']] = text.sentiment.subjectivity

    def extract_initial_process_features(self, sentences_list, train=False):
        #TODO: consider removeing stop words
        all_words_list = []
        interjection_dict = {}
        intersifier_dict = {}
        intersifier_pos_count, interjection_pos_count = 0,0
        intersifier_neg_count, interjection_neg_count = 0,0
        # self.words_dic = {}
        adverbs_count = 0
        adjective_count = 0
        interjections_count = 0
        for i, sentence in enumerate(sentences_list):
            processed_article = re.sub('[^a-zA-Z]', ' ', sentence)
            processed_article = re.sub(r'\s+', ' ', processed_article)
            words_list = nltk.word_tokenize(processed_article) 
            
            word_tags = nltk.pos_tag(words_list)
            adjective = 0
            adverb = 0
            interjection = 0
            positive_word = 0
            negative_word = 0

            for w, tag in word_tags:
                if tag in ["JJ", "JJR", "JJS"]:
                    adjective = 1
                elif tag in ["RR", "RBR", "RBS"]:
                    adverb = 1
                elif w in self.interjections:
                    interjection = 1

                word_blob = TextBlob(w)
                if word_blob.sentiment.polarity >= 0.5:
                    positive_word=1
                elif word_blob.sentiment.polarity <= -0.5:
                    negative_word=1
            
            adverbs_count += adverb
            adjective_count+=adjective
            interjections_count +=interjection
            self.features_matrix[i,self.features_map['adjectives']] = adjective
            self.features_matrix[i,self.features_map['adverbs']] = adverb
            self.features_matrix[i,self.features_map['interjections']] = interjection
            self.features_matrix[i,self.features_map['positive_word']] = positive_word
            self.features_matrix[i,self.features_map['negative_word']] = negative_word
            


            # intersifier_count, interjection_count = 0,0
            # for word in words_list:
            #     if word in self.words_dic.keys():
            #         if self.y_train[i]:
            #             self.words_dic[word]['pos']+=1
            #         else:
            #             self.words_dic[word]['neg']+=1
            #     else:
            #         if self.y_train[i]:
            #             self.words_dic[word] = {
            #                     'pos': 1,
            #                     'neg': 0
            #                 }
            #         else:
            #             self.words_dic[word] = {
            #                     'pos': 0,
            #                     'neg': 1
            #                 }

                # if word in self.intensifiers:
                #     intersifier_count+=1
                #     if word in intersifier_dict.keys():
                #         if self.y_train[i]:
                #             intersifier_dict[word]['pos'] += 1
                #             intersifier_pos_count+=1
                #         else:
                #             intersifier_dict[word]['neg'] += 1
                #             intersifier_neg_count+=1
                #     else:
                #         if self.y_train[i]:
                #             intersifier_dict[word] = {
                #                 'pos': 1,
                #                 'neg': 0
                #             }
                #             intersifier_pos_count+=1
                #         else:
                #             intersifier_dict[word] = {
                #                 'pos': 0,
                #                 'neg': 1
                #             }
                #             intersifier_neg_count+=1
                # if word in self.interjections:
                #     interjection_count+=1
                #     if word in interjection_dict.keys():
                #         if self.y_train[i]:
                #             interjection_dict[word]['pos'] += 1
                #             interjection_pos_count+=1
                #         else:
                #             interjection_dict[word]['neg'] += 1
                #             interjection_neg_count+=1
                #     else:
                #         if self.y_train[i]:
                #             interjection_dict[word] = {
                #                 'pos': 1,
                #                 'neg': 0
                #             }
                #             interjection_pos_count+=1
                #         else:
                #             interjection_dict[word] = {
                #                 'pos': 0,
                #                 'neg': 1
                #             }
                #             interjection_neg_count+=1
        

            # self.features_matrix[i,self.features_map['intersifier']] = intersifier_count
            # self.features_matrix[i,self.features_map['interjection']] = interjection_count
            self.features_matrix[i,self.features_map['sentence_length']] = len(words_list)
            all_words_list.append(words_list)       

        self.tfidf_matrix = create_tfidf(all_words_list)
        print("Adverbs: {}".format(adverbs_count))
        print("Adjectives: {}".format(adjective_count))
        print("Interjections: {}".format(interjections_count))
'''