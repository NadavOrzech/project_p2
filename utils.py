import json
import os
import glob

import matplotlib.pyplot as plt


def create_parse_dataset():
    input_path = "C:\\Users\\Dell\\Desktop\\Technion\\Project\\news-headlines-dataset-for-sarcasm-detection\\dataset_part2_with_scores.json"
    output_path = "C:\\Users\\Dell\\Desktop\\Technion\\Project\\news-headlines-dataset-for-sarcasm-detection\\dataset_part2_no_article.json"

    with open(input_path, 'r') as input_json:
        data = json.load(input_json)

    out_data = {}
    for key in data:
        headline = data[key]['headline']
        description = data[key]['short_description']
        author = data[key]['author']
        out_data[key] = {
            "headline": headline,
            'short_description': description,
            'author': author
        }

    with open(output_path,'w') as out_json:
        json.dump(out_data,out_json)
    aaa=2

def create_knn_cv_plot():
    input_path = "C:\\Users\\Dell\\Desktop\\Technion\\Project\\source\\project\\KNN_cv.json"
    with open(input_path, 'r') as input_json:
        data = json.load(input_json)
        data = data['[0, 1, 2, 5, 6, 8, 9]']

    uniform = []
    distance = []
    x_uni = []
    x_dist = []

    for key in data:
        k = int(key.split(',')[0][1:])

        if 'uniform' in key:
            x_uni.append(k)
            uniform.append(data[key]['mean'])
        else:
            x_dist.append(k)
            distance.append(data[key]['mean'])

    assert x_uni == x_dist

    plt.plot(x_uni,uniform, linewidth=4, label="uniform")
    plt.plot(x_dist,distance, linewidth=2, label="distance")
    # plt.plot(x,avg_values,label="average")
    plt.xlabel('K value')
    # plt.xticks(x,x)
    plt.ylabel('Accuracy [%]')
    plt.title('Cross Validation across K-vals')
    plt.legend()
    plt.grid()
    plt.savefig("KNN_CV")
    aaaa=2



def create_dct_cv_plot():
    input_path = "C:\\Users\\Dell\\Desktop\\Technion\\Project\\source\\project\\DCT_cv.json"
    with open(input_path, 'r') as input_json:
        data = json.load(input_json)
        data = data["[0, 1, 3, 5, 6, 7, 8, 9]"]

    min_samples_2 = []
    min_samples_5 = []
    min_samples_10 = []
    min_samples_20 = []

    x_2 = []
    x_5 = []
    x_10 = []
    x_20 = []

    for key in data:
        min_split = int(key.split(',')[0].split(':')[1])
        min_leaf = int(key.split(',')[1].split(':')[1])

        if min_split == 2:
            x_2.append(min_leaf)
            min_samples_2.append(data[key]['mean'])
        elif min_split == 5:
            x_5.append(min_leaf)
            min_samples_5.append(data[key]['mean'])
        elif min_split == 10:
            x_10.append(min_leaf)
            min_samples_10.append(data[key]['mean'])
        elif min_split == 20:
            x_20.append(min_leaf)
            min_samples_20.append(data[key]['mean'])
            
    assert x_2 == x_5 == x_10 == x_20

    plt.plot(x_2, min_samples_2, label="min_samples_split = 2")
    plt.plot(x_5, min_samples_5, label="min_samples_split = 5")
    plt.plot(x_10, min_samples_10, label="min_samples_split = 10")
    plt.plot(x_20, min_samples_20, label="min_samples_split = 20")

    plt.xlabel('Min Samples For Leaf')
    # plt.xticks(x,x)
    plt.ylabel('Accuracy [%]')
    plt.title('Cross Validation across min samples for split and leaf')
    plt.legend()
    plt.grid()
    plt.savefig("DCT_CV")


def create_svm_cv_plot():
    input_path = "C:\\Users\\Dell\\Desktop\\Technion\\Project\\source\\project\\SVM_cv.json"
    with open(input_path, 'r') as input_json:
        data = json.load(input_json)
        data = data["[0, 1, 2, 5, 6, 8, 9]"]

    linear = []
    rbf = []
    poly = []
    sigmoid = []

    x_linear = []
    x_rbf = []
    x_poly = []
    x_sigmoid = []

    for key in data:
        kernel = key.split(',')[0].split(': ')[1]
        c = float(key.split(',')[1].split(': ')[1])

        if kernel == 'linear':
            x_linear.append(c)
            linear.append(data[key]['mean'])
        elif kernel == 'rbf':
            x_rbf.append(c)
            rbf.append(data[key]['mean'])
        elif kernel == 'poly':
            x_poly.append(c)
            poly.append(data[key]['mean'])
        elif kernel == 'sigmoid':
            x_sigmoid.append(c)
            sigmoid.append(data[key]['mean'])
    
    assert x_linear == x_rbf == x_poly == x_sigmoid

    plt.plot(x_linear, linear, label="linear")
    plt.plot(x_rbf, rbf, label="rbf")
    plt.plot(x_poly, poly, label="poly")
    plt.plot(x_sigmoid, sigmoid, label="sigmoid")

    plt.xlabel('C hyperparameter')
    # plt.xticks(x,x)
    plt.ylabel('Accuracy [%]')
    plt.title('Cross Validation across kernels and C vals')
    plt.legend()
    plt.grid()
    plt.savefig("SVM_CV")

input_path = "C:\\Users\\Dell\\Desktop\\Technion\\Project\\source\\project\\RF_cv.json"
with open(input_path, 'r') as input_json:
    data = json.load(input_json)
    data = data["[0, 1, 4, 5, 7, 8, 9]"]

min_samples_2 = []
min_samples_5 = []
min_samples_10 = []
min_samples_20 = []

x_2 = []
x_5 = []
x_10 = []
x_20 = []

for key in data:
    n_estimators = int(key.split(',')[0].split(':')[1])
    min_samples = int(key.split(',')[1].split(':')[1])

    if min_samples == 2:
        x_2.append(n_estimators)
        min_samples_2.append(data[key]['mean'])
    elif min_samples == 5:
        x_5.append(n_estimators)
        min_samples_5.append(data[key]['mean'])
    elif min_samples == 10:
        x_10.append(n_estimators)
        min_samples_10.append(data[key]['mean'])
    elif min_samples == 20:
        x_20.append(n_estimators)
        min_samples_20.append(data[key]['mean'])

assert x_2 == x_5 == x_10 == x_20

plt.plot(x_2, min_samples_2, label="min_samples_split = 2")
plt.plot(x_5, min_samples_5, label="min_samples_split = 5")
plt.plot(x_10, min_samples_10, label="min_samples_split = 10")
plt.plot(x_20, min_samples_20, label="min_samples_split = 20")

plt.xlabel('Number of Estimators hyperparameter')
# plt.xticks(x,x)
plt.ylabel('Accuracy [%]')
plt.title('Cross Validation across Num of Estimators and Min Samples Split')
plt.legend()
plt.grid()
plt.savefig("RF_CV")


aaaa=2

