import training
import sklearn
# from data_preproccess import create_data_list, create_tfidf
from dataloader import Dataloader
from config import Config
from itertools import chain, combinations
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns

TEST_SIZE = 0.15
RANDOM_STATE = 42

def calc_heat_map(data1, data2):
    feature_hist = [
        [[],[],[],[],[],[],[],[],[],[]],
        [[],[],[],[],[],[],[],[],[],[]],
        [[],[],[],[],[],[],[],[],[],[]],
        [[],[],[],[],[],[],[],[],[],[]],
        [[],[],[],[],[],[],[],[],[],[]],
        [[],[],[],[],[],[],[],[],[],[]],
        [[],[],[],[],[],[],[],[],[],[]],
        [[],[],[],[],[],[],[],[],[],[]],
        [[],[],[],[],[],[],[],[],[],[]],
        [[],[],[],[],[],[],[],[],[],[]],
        ]
    datas = [data1,data2]
    for data in datas:
        for key in data:
            for f in range(10):
                if str(f) not in key:
                    continue
                for k in data[key]:
                    acc = data[key][k]['mean']
                    for f2 in range(10):
                        if f==f2 or str(f2) not in key:
                            continue
                        feature_hist[f][f2].append(acc)
    res = []
    for i in range(10):
        res_i = []
        for j in range(10):
            acc_list = feature_hist[i][j]
            acc = 0
            if len(acc_list) > 0:
                acc = sum(acc_list)/len(acc_list)
            res_i.append(acc)
        res.append(res_i)

    ax = sns.heatmap(res,annot=True)
    ax.figure.savefig('./a.jpg')
    aaa=2

    
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

if __name__ == "__main__":
    input_file = "C:\\Users\\Dell\\Desktop\\Technion\\Project\\source\\project\\\\svm_cv\\SVM_cv_4.json"
    with open(input_file,'r') as in_json:
        data1 = json.load(in_json)
    input_file = "C:\\Users\\Dell\\Desktop\\Technion\\Project\\source\\project\\svm_cv\\SVM_cv_10.json"
    with open(input_file,'r') as in_json:
        data2 = json.load(in_json)
    # input_file = "C:\\Users\\Dell\\Desktop\\Technion\\Project\\source\\project\\DCT_cv.json"
    # with open(input_file,'r') as in_json:
    #     data1 = json.load(in_json)

    # calc_heat_map(data1,data2)
    feature_hist = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
    }
    for key in data1:
        for f in range(10):
            if str(f) in key:
                continue
            acc = data1[key]['mean']
            feature_hist[f].append(acc)
                
    for key in data2:
        for f in range(10):
            if str(f) in key:
                continue
            acc = data2[key]['mean']
            feature_hist[f].append(acc)

    x = list(range(10))
    min_values = [min(feature_hist[i])*100 for i in range(10)]
    max_values = [max(feature_hist[i])*100 for i in range(10)]
    avg_values = [sum(feature_hist[i])/len(feature_hist[i])*100 for i in range(10)]   

    plt.plot(x,min_values,label="min")
    plt.plot(x,max_values,label="max")
    plt.plot(x,avg_values,label="average")
    plt.xlabel('Feature Index')
    plt.xticks(x,x)
    plt.ylabel('Accuracy [%]')
    plt.title('Subset of Features Excluding Feature X')
    plt.legend()
    plt.grid()
    plt.savefig("subset_excluding_feature")


    exit(0)
    config = Config()

    dataloader = Dataloader(config,generate_tfidf=False,feature_flag=True)
    
    x_train, y_train = dataloader.get_train_dataloader()
    
    # model = KNeighborsClassifier(n_neighbors=1001,weights='distance')
    # scores = cross_val_score(model, x_train[:,[0,1,2,5,6,8,9]], y_train, cv=10)

    knn_cv_dict = {
        'model_name': 'KNN',
        'k_list': [1,3,5,7,9,11,13,15,17,19,21,25,31,41,51,61,71,91,121,151,181,211,251,301,351,401,501,601,801,1001]
    }

    svm_cv_dict = {
        'model_name': 'SVM',
        'c_list': [0.5,1,2,4],
        'kernel_list': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    decision_tree_cv_dict = {
        'model_name': 'DCT',
        'min_samples_split': [1,2,5,10]
    }
    decision_tree_cv_dict = {
        'model_name': 'RF',
        'min_samples_split': [1,2,5,10]
    }
    # scores = training.cross_validation(x_train,y_train,knn_cv_dict,features_permute=True,specific_subset=[0,1,2,5,6,8,9])
    # scores = training.cross_validation(x_train,y_train,decision_tree_cv_dict,features_permute=True,powerset_size=list(range(11)))
    
    scores = training.cross_validation(x_train,y_train,svm_cv_dict,features_permute=True,powerset_size=[5,6,7,8,9,10])
    # scores = training.cross_validation(x_train,y_train,decision_tree_cv_dict,features_permute=True,powerset_size=list(range(11)))

    aaa=2