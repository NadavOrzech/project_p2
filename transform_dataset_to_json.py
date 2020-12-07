import json
import os 
import glob

import nltk
import gensim

input_file = "C:\\Users\\Dell\\Desktop\\Technion\\Project\\news-headlines-dataset-for-sarcasm-detection\\Sarcasm_Headlines_Dataset_v2.json"


data = {}

f = open(input_file, 'r')

tmp_list = f.readlines()
key = 0
max_length = 0
for idx in range(0, len(tmp_list)-1, 4):
    is_sarcastic = int(tmp_list[idx+1].split(':')[1].split(',')[0])
    headline = tmp_list[idx+2].split('"headline": ')[1].split(',\n')[0][1:-1]
    article_link = tmp_list[idx+3].split("\"")[3]
    assert tmp_list[idx+3].count("\"") == 4
    curr_data = {
        "is_sarcastic": is_sarcastic,
        "headline": headline,
        "article_link": article_link
    }
    if len(headline.split(' ')) >= 34:
        continue
    max_length = max(max_length, len(headline.split(' ')))
    data[key] = curr_data
    key += 1
    aa = 2
print(max_length)

with open(os.path.join(os.path.dirname(input_file), 'sarcasm_dataset_v3.json'), 'w') as output:
    json.dump(data, output)