import json
import os
import glob

from BertClassifier import BertClassifier
from config import Config

input_dir = 'C:\\Users\\Dell\\Desktop\\Technion\\Project\\articles_part2'
files_list = glob.glob(os.path.join(input_dir,'*'))
config = Config()

bert_classifier = BertClassifier(None,None,config=config, checkpoint_file="bert_classifier")

for f in files_list:
    with open(f, 'r', encoding='utf-8', ) as json_file:
        data = json.load(json_file)

        for key in data:
            article = data[key]['article']
            sentence_list = article.split('.')
            labels_list = []
            for sent in sentence_list:
                sent = sent.lower()
                res = bert_classifier.inference(sent)
                labels_list.append(res)
            
            data[key]['score'] = labels_list
    
    res_file = os.path.basename(f).split('.json')[0] + '_new.json'
    res_file = os.path.join('.',res_file)
    with open(res_file, 'w', encoding='utf-8', ) as json_res:
        json.dump(data,json_res)
    aaaa=2