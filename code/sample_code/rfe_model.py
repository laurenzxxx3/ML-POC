#!/usr/bin/python3

"""
A Recursive Feature Elimination model, build for implementation in final_script.py
"""

# Imports
import sys
import pandas as pd
import numpy  as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import statistics
import collections

# RFE
def rfe(data):
    # Seperate data in labels and features
    features = data.drop('Subgroup', axis =1)
    labels = data.Subgroup
    # Define dictionary to save final results
    final_result = {}
    # Set range of random states to be used by train_test_split
    random_states = range(100)
    # "Outer loop" of 100 iterations, with 10fold cross validation done for each iteration
    for random_state in random_states:
        # Splits data into train and test features with a different random state at each iteration
        train_features, test_features, train_labels, test_labels=train_test_split(features, labels, test_size = 0.1, random_state = random_state)
        # Set parameters to be used for GridSearch
        param_grid = [{'estimator__C': [0.0001,0.001,0.01,0.1,1.0]}]
        # Create SCV estimator to be used to measure performance
        estimator = SVC(kernel="linear",class_weight='balanced')
        # Create the RFE object and compute a cross-validated score
        selector = RFECV(estimator=estimator, step=1, min_features_to_select=1, cv=10,
                      scoring='accuracy',n_jobs=-1)
        # Create GridSearch object
        clf = GridSearchCV(selector, param_grid, cv=10)
        # Fit GridSearch object to data
        clf.fit(train_features, train_labels)
        # Put the best features into new df
        features_new = clf.best_estimator_.transform(train_features)
        # save best C parameter out of inner cv loop and list of used features
        best_svm_param = clf.best_params_['estimator__C']
        used_features = train_features.columns[clf.best_estimator_.support_]
        # train model on entire train set with best parameters and features
        model = SVC(kernel="linear",class_weight='balanced',C=best_svm_param)
        model.fit(train_features[used_features], train_labels)
        # obtain accuracy of model on train and test set
        test_acc = sum(test_labels == model.predict(test_features[used_features]))/float(len(test_labels))
        train_acc = sum(train_labels == model.predict(train_features[used_features]))/float(len(train_labels))
        # save result in a dictionary
        if best_svm_param in final_result.keys():
            final_result[best_svm_param]["test_acc"].append(test_acc)
            final_result[best_svm_param]["train_acc"].append(train_acc)
            final_result[best_svm_param]["n_features"].append(len(used_features))
            final_result[best_svm_param]["used_features"].append(used_features)
            final_result[best_svm_param]["times param used"]+=1
            final_result[best_svm_param]["common features"]=set.intersection(*[set(list) for list in final_result[best_svm_param]["used_features"]])
        else:
            final_result[best_svm_param] = {}
            final_result[best_svm_param]["test_acc"] = [test_acc]
            final_result[best_svm_param]["train_acc"] =[train_acc]
            final_result[best_svm_param]["n_features"]=[len(used_features)]
            final_result[best_svm_param]["used_features"]= [used_features]
            final_result[best_svm_param]["times param used"]=1
    times_param_used=0
    # record used features for best parameter
    for param in final_result:
        if final_result[param]["times param used"]>times_param_used:
            times_param_used=final_result[param]["times param used"]
            final_svm_param=param
    rfe_feature_list = final_result[final_svm_param]["used_features"]
    # combine used features into a single list
    rfe_list=[]
    for list in rfe_feature_list:
        for i in list:
            rfe_list.append(i)
    # reduce used features to optimal features
    counter_list=collections.Counter(rfe_list)
    sorted_list=[]
    for i in counter_list:
        if counter_list[i]>times_param_used/2:
            sorted_list.append(i)
    # calculate mean test accuracy for best parameter
    mean_accuracy=statistics.mean(final_result[final_svm_param]["test_acc"])
    # print final results
    print("best C parameter:",final_svm_param)
    print("times parameter used:",times_param_used)
    print("mean accuracy:",mean_accuracy)
    print("optimal features:",sorted_list)
    #return final_result
