import os
import json
import urllib
import requests
from six.moves.urllib.request import urlopen
from urllib.request import Request, urlopen
from html.parser import HTMLParser
from bs4 import BeautifulSoup
import unicodedata
import io

input_file = 'C:\\Users\\Dell\\Downloads\\News_Category_Dataset_v2.json\\News_Category_Dataset_v2.json'
input_file = "C:\\Users\\Dell\\Desktop\\Technion\\Project\\news-headlines-dataset-for-sarcasm-detection\\dataset_part2.json"
# output_file = 'C:\\Users\\Dell\\Downloads\\News_Category_Dataset_v2.json\\new_data.json'
# link = "https://www.huffingtonpost.com/entry/texas-amanda-painter-mass-shooting_us_5b081ab4e4b0802d69caad89"
output_dir = 'C:\\Users\\Dell\\Desktop\\Technion\\Project\\articles_part2'

def retrieve_article_text(link):
    req = Request(link, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        webpage = urlopen(req).read()
    except:
        return ' '
    soup = BeautifulSoup(webpage,"html.parser")
    name_box = soup.findAll("p")
    p_list = [p.text.strip() for p in name_box]
    text = ''
    for p in p_list:
        # if '\xa0' in p:
        #     # p_tmp=p.rem
        #     print(p.replace('\xa0', ' '))
        text+=p+' '
        # text+=p.replace('\xa0', ' ')+' '

    return text
    # return unicodedata.normalize("NFKC", text)

with open(input_file, 'r') as file_json:
    data = json.load(file_json)

article_data={}
for key in data:
    # print(key)
    example = data[key]
    head = example['headline']
    link = example['link']
    text = retrieve_article_text(link)
    article_data[key]={
        'headline': head,
        'article': text,
        'score': 0
    }
    if int(key)>0 and int(key) % 1000 == 0:
        path = os.path.join(output_dir,'file_{}.json'.format(int(int(key)/1000)))
        with open(path,'w',encoding="utf-8") as out_json:
            json.dump(article_data,out_json,ensure_ascii=False)
        article_data={}
aaa=1


