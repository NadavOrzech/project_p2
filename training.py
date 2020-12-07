import sklearn
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score
from itertools import chain, combinations

import json
import os

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def cross_validation(X_train, y_train, cv_params, features_permute=False,powerset_size=None, specific_subset=None):
    total_scores = {}
    features_list = [list(range(X_train.shape[1]))]
    if powerset_size is None:
        powerset_size=[X_train.shape[1]]
    if features_permute:
        features_list = list(powerset(range(X_train.shape[1])))
    if cv_params['model_name'] == "KNN":
        if specific_subset is None:
            for features_comb in features_list[1:]:
                if len(features_comb) not in powerset_size:
                    continue
                total_scores[str(features_comb)] = {}
                for k in cv_params['k_list']:
                    model = KNeighborsClassifier(n_neighbors=k)
                    scores = cross_val_score(model, X_train[:,features_comb], y_train, cv=10)
                    total_scores[str(features_comb)]['k{}'.format(k)] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                    }
        else:
            features_comb = specific_subset
            total_scores[str(features_comb)] = {}
            for k in cv_params['k_list']:
                model = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(model, X_train[:,features_comb], y_train, cv=10)
                total_scores[str(features_comb)]['k{}'.format(k)] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                }
                    # break

    elif cv_params['model_name'] == "SVM":
        l=5
        for features_comb in features_list[1:]:
            if len(features_comb) not in powerset_size:
                continue
            print("calculating for "+str(features_comb))
            total_scores[str(features_comb)] = {}
            if False:
                for kernel in cv_params['kernel_list']:
                    for c in cv_params['c_list']:
                        model = SVC(C=c,kernel=kernel)
                        scores = cross_val_score(model, X_train[:,features_comb], y_train, cv=10)
                        total_scores[str(features_comb)]['kernel: {}, C: {}'.format(kernel,c)] = {
                            'mean': scores.mean(),
                            'std': scores.std(),
                            # 'comb': features_comb,
                        }
            else:
                model = SVC(kernel='linear')
                scores = cross_val_score(model, X_train[:,features_comb], y_train, cv=10)
                total_scores[str(features_comb)]= {
                            'mean': scores.mean(),
                            'std': scores.std(),
                            }
            if len(features_comb) > l:
                with open(os.path.join('.','{}_cv_{}.json'.format(cv_params['model_name'],len(features_comb))), 'w') as json_file:
                    json.dump(total_scores,json_file)
                l+=1

    elif cv_params['model_name'] == "DCT":
        for features_comb in features_list[1:]:
            if len(features_comb) not in powerset_size:
                continue
            total_scores[str(features_comb)] = {}
            if False:
                for min_split in cv_params['min_samples_split']:
                    model = DecisionTreeClassifier(min_samples_split=min_split)
                    scores = cross_val_score(model, X_train[:,features_comb], y_train, cv=10)
                    total_scores[str(features_comb)]['min_samples_split: {}'.format(min_split)] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        # 'comb': features_comb,
                       }

            else:
                model = DecisionTreeClassifier()
                scores = cross_val_score(model, X_train[:,features_comb], y_train, cv=10)
                total_scores[str(features_comb)]= {
                            'mean': scores.mean(),
                            'std': scores.std(),
                            }
 
    elif cv_params['model_name'] == "RF":
        for features_comb in features_list[1:]:
            if len(features_comb) not in powerset_size:
                continue
            total_scores[str(features_comb)] = {}
            if False:
                # for min_split in cv_params['min_samples_split']:
                #     model = RandomForestClassifier(n_estimators=)
                #     scores = cross_val_score(model, X_train[:,features_comb], y_train, cv=10)
                #     total_scores[str(features_comb)]['min_samples_split: {}'.format(min_split)] = {
                #         'mean': scores.mean(),
                #         'std': scores.std(),
                #        }
                pass
            else:
                model = RandomForestClassifier()
                scores = cross_val_score(model, X_train[:,features_comb], y_train, cv=10)
                total_scores[str(features_comb)]= {
                            'mean': scores.mean(),
                            'std': scores.std(),
                            }
 


    with open(os.path.join('.','{}_cv.json'.format(cv_params['model_name'])), 'w') as json_file:
        json.dump(total_scores,json_file)

    return total_scores


def bernoulli_model(X_train, X_test, y_train, y_test):
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    y_pred = bnb.predict(X_test)
    accuracies = accuracy_score(y_test, y_pred)
    print(f"Bernoulli Accuracy: {accuracies}")


def KNN_model(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies = accuracy_score(y_test, y_pred)
    print(f"KNN Accuracy: {accuracies}")


def SVM_model(X_train, X_test, y_train, y_test):
    svm = SVC(kernel="linear")
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracies = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {accuracies}")


def decision_tree_model(X_train, X_test, y_train, y_test):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    accuracies = accuracy_score(y_test, y_pred)
    print(f"decision_tree Accuracy: {accuracies}")


def random_forest_model(X_train, X_test, y_train, y_test):
    rnd_forest = RandomForestClassifier(n_estimators=50)
    rnd_forest.fit(X_train, y_train)
    y_pred = rnd_forest.predict(X_test)
    accuracies = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracies}")
