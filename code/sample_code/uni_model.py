#!/usr/bin/python3

"""
A univariate data analysis model, build for implementation in final_script.py
"""

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2


# Split data:
def splitter(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
    return (X_train, X_test, y_train, y_test)

# Build confusion matrix and determine Accuracy
def build_cm(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return (cm, accuracy)

# Performs univariate chi2 test
def univariate_test(x_train, y_train, n):
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    X = scaler.transform(x_train)
    y = y_train
    n_features = n
    SelectFeatures = SelectKBest(chi2, k=n_features)
    SelectFeatures.fit(X, y)
    Features2 = SelectFeatures.get_support(indices=True)
    chi_2_features = x_train[Features2]
    return (chi_2_features, Features2)

# Basic Linear SVM
def Linear_svm(X_train, Y_train, X_test):
    clf = SVC(kernel="linear", gamma='auto')
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    return (clf, y_pred)


def uni(data):
    # Load features and labels
    features = data.drop('Subgroup', axis =1)
    labels = data.Subgroup
    # select optimal number of features between 2 and 100 in kfold test/train split
    Test_list = []
    testing_dict={}
    feature_dict={}
    folds = 10
    # Loop nfolds over test/train split to determine optimal number of features.
    for i in range(folds):
        X_train, X_test, y_train, y_test = splitter(features, labels)
        for k in range(2,101):
            chi2_features, Features = univariate_test(X_train, y_train, k)
            X_test_features = X_test.T.iloc[Features]

            clf, y_pred = Linear_svm(chi2_features, y_train, X_test_features.T)
            cm, Accuracy = build_cm(y_test, y_pred)
            if k not in testing_dict.keys():
                testing_dict[k] = Accuracy/folds
            else:
                testing_dict[k] += Accuracy/folds

    # Determine best amount of features
    maxvalue = 0
    for k in range(2,len(testing_dict)+2):
        if testing_dict[k] > maxvalue:
            maxvalue = testing_dict[k]
            maxkvalue = k+2

    print("Highest accuracy: ", maxvalue)
    print("Amount of features:", maxkvalue)

    # Select features from whole dataset
    chi2_features, Features = univariate_test(features, labels, maxkvalue)
    accuracy_list = []

    # Determine accuracy in validation:
    for i in range(10):
        X_train, X_test, y_train, y_test = splitter(features, labels)
        X_test_features = X_test.T.iloc[Features]
        X_train = X_train.T.iloc[Features]
        clf, y_pred = Linear_svm(X_train.T, y_train, X_test_features.T)
        cm, Accuracy = build_cm(y_test, y_pred)
        accuracy_list.append(Accuracy)

    test_array = np.array(accuracy_list)

    print("Average Accuracy: ", test_array.mean())
    print("Standard deviation: ", test_array.std())
    print("Features used: ", Features)
