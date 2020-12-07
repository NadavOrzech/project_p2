import json
import os
import glob



raw_file_path = 'C:\\Users\\Dell\\Desktop\\Technion\\Project\\news-headlines-dataset-for-sarcasm-detection\\dataset_part2.json'
after_bert_dir = 'C:\\Users\\Dell\\Desktop\\Technion\\Project\\articles_part2\\after_bert'
num_of_file = 36

final_data = {}
final_data_index = 0
max_score = 0
with open(raw_file_path, 'r',encoding='utf-8') as raw_json:
    raw_data = json.load(raw_json)

for i in range(num_of_file):

    new_file_path = os.path.join(after_bert_dir, 'file_{}_new.json'.format(i+1))
    

    with open(new_file_path, 'r',encoding='utf-8') as new_json:
        new_data = json.load(new_json)

    for key in new_data:
        headline = new_data[key]['headline']
        found = False
        for r_key in raw_data:
            if raw_data[r_key]['headline'] == headline:
                author = raw_data[r_key]['authors']
                category = raw_data[r_key]['category']
                short_description = raw_data[r_key]['short_description']
                date = raw_data[r_key]['date']

                article = new_data[key]['article']   
                scores_vec = new_data[key]['score']
                score = sum(scores_vec)/len(scores_vec)          
                
                final_data[final_data_index]={
                    'score': score,
                    'headline': headline,
                    'author': author,
                    'category': category,
                    'date': date,
                    'short_description': short_description,
                    'article': article,
                    'scores_vector': scores_vec,
                }
                max_score = max(max_score, score)
                final_data_index+=1
                found = True
                break
        if not found:
            print('not found')
            

    aaa=2

for f_key in final_data:
    score = final_data[f_key]['score']
    norm_score = score/max_score
    if norm_score>1:
        tmp=2
    final_data[f_key]['score'] = norm_score

output_file = os.path.join('C:\\Users\\Dell\\Desktop\\Technion\\Project\\news-headlines-dataset-for-sarcasm-detection','dataset_part2_with_scores.json')
with open(output_file,'w') as out_json:
    json.dump(final_data,out_json)
aaa=1