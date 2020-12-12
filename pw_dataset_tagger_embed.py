import json
import os
import glob
import nltk
import torch
# from BertClassifier import BertClassifier
from config import Config
from BertEmbedder import BertEmbedder
input_dir = '.' 
files_list = glob.glob(os.path.join(input_dir, 'dataset_part2_no_article.json'))
config = Config()

# bert_classifier = BertClassifier(None, None, config=config, checkpoint_file="bert_classifier")
headlines = []
descriptions = []
authors =[]
authors_name = {}
i = 0
output_data = {}
for f in files_list:

    bert_embedder = BertEmbedder(headlines, [], config)

    with open(f, 'r', encoding='utf-8', ) as json_file:
        data = json.load(json_file)
        for data_i, key in enumerate(data):
            headline = data[key]['headline'].lower()
            description = data[key]['short_description'].lower()
            author = data[key]['author'].lower()
            if author not in authors_name.keys():
                authors_name[author] = i
                i += 1


            embeddings_headlines = bert_embedder.get_single_sentence_embedding(40, headline)
            embeddings_descriptions = bert_embedder.get_single_sentence_embedding(95, description)

            output_data[key] = {
                'headline': headline,
                'short_description': description,
                'author': author,
                'headline_embedding': embeddings_headlines,
                'description_embedding': embeddings_descriptions,
                'author_embedding': authors_name[author]
            }

            if data_i%1000 == 0:
                print("generates file number: {}".format(i/1000))
                res_file = os.path.basename(f).split('.json')[0] + '_{}_embeddings.json'.format(int(data_i/1000))
                res_file = os.path.join('.', res_file)
                with open(res_file, 'w', encoding='utf-8', ) as json_res:
                    json.dump(output_data, json_res)
