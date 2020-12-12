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


class DataloaderPart2():
    def __init__(self, params):
        self.input_path = params.part2_data_path
        with open(self.input_path,'r') as input_json:
           self.data = json.load(input_json)

        second_input = "C:\\Users\\Dell\\Desktop\\Technion\\Project\\news-headlines-dataset-for-sarcasm-detection\\dataset_part2_with_scores_sarcasm_labels.json"
        with open(second_input,'r') as input_json2:
            self.data2 = json.load(input_json2)

        for key in self.data:
            head = self.data[key]['headline']
            found = False
            for key2 in self.data2:
                if self.data2[key2]['headline'] == head:
                    des_score = self.data2[key2]['description_score']
                    head_score = self.data2[key2]['headline_score']
                    self.data[key]['description_score'] = des_score
                    self.data[key]['headline_score'] = head_score
                    found = True
                    break
            if not found:
                print("NOT FOUND")
        
        output = "C:\\Users\\Dell\\Desktop\\Technion\\Project\\news-headlines-dataset-for-sarcasm-detection\\dataset_part2_with_scores_and_lables.json"
        with open(output,'w') as output_json:
           json.dump(self.data,output_json)

        self.category_list = params.part2_category_list
        self.num_samples = len(self.data)
        self.features_map = params.part2_features_map

    def generate_features_matrix(self):
        data = self.data
        features_matrix = np.zeros((self.num_samples,len(self.features_map)))
        for i, key in enumerate(data):
            category = data[key]['category']
            date = data[key]['date'].split('-')
            year = date[0]
            month = date[1]
            day = date[2]


            features_matrix[i,self.features_map['category']] = self.category_list[category]
            features_matrix[i,self.features_map['year']] = year
            features_matrix[i,self.features_map['month']] = month
            features_matrix[i,self.features_map['day']] = day
            


        aaaa=2


if __name__ == "__main__":
    params = Config()
    dataloader = DataloaderPart2(params)
    dataloader.generate_features_matrix()