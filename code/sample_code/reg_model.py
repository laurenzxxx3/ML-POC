#!/usr/bin/python3

"""
A Elastic net regularization model, build for implementation in final_script.py
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

# calculates prediction accuracy( requires the prediction to be appended to end of dataset in a column names "predicted")
# also requires model to have "predict parameter"
def accuracy(data_set, model):
    a = data_set[["Subgroup"]].copy()
    a["predicted"] = model.predict(data_set.loc[:,data_set.columns != "Subgroup"])
    acc_i = sum(a.Subgroup == a.predicted)/float(len(a))
    acc_s = str(sum(a.Subgroup == a.predicted)) + "/"+ str(len(a))
    return( acc_i)



def elasticnet(data):
    # dict to safe results in
    final_result= {}
    # Split data into features and albels
    features = data.drop('Subgroup', axis =1)
    labels = data.Subgroup
    # setting up data splits for the outer CV loop
    for r in range(100):
        outer_cv = StratifiedShuffleSplit(10, 0.1)
        for train_index, test_index in outer_cv.split(features,labels):
                train_set = data.iloc[train_index]
                test_set = data.iloc[test_index]
                # set up empty data frames for final predictions
                train_predictions = pd.DataFrame(index=train_set.index)
                train_predictions["Subgroup"] = train_set.Subgroup
                train_predictions["predicted"] = None
                test_predictions = pd.DataFrame(index=test_set.index)
                test_predictions["Subgroup"] = test_set.Subgroup
                test_predictions["predicted"] = None

                ### inner cross validaiton loop - hyperparameter performance

                # set model parameters(SVM with elastic net regression)
                model = SGDClassifier(penalty="elasticnet", max_iter = (10**6/len(train_set)), tol=10**-3, n_jobs=-2)
                # define hyperparameter to test in inner cross validation
                parameters = {"alpha": 10.0 ** -np.arange(1, 7),
                              "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
                # create cv object(look into fine tuning)
                cv = GridSearchCV(model, parameters, cv=2, scoring="accuracy", return_train_score=False, n_jobs=-2)
                # run cross validation on training set with specified parameters
                cv.fit(train_set.loc[:, train_set.columns != "Subgroup"], train_set.Subgroup)
                # from here we are selecting the top # parameters combinations instead of the single best due to some parameter settings
                # performing (equally) good while using less features.
                result = pd.DataFrame(cv.cv_results_)
                ranked_results = result.sort_values(by="rank_test_score")
                ### train models on entire training set(of this split) with parameters of top performing settings of inner CV loop
                ### and save all results to a dict
                #iterate over top 10 hyper parameter combinations
                for i in range(len(ranked_results[0:10])):
                    rank = list(ranked_results.rank_test_score)[i]
                    alpha = list(ranked_results.param_alpha)[i]
                    l1r = list(ranked_results.param_l1_ratio)[i]
                    # train model with specified hyper parameters
                    model = SGDClassifier(penalty="elasticnet", max_iter= (10**6/len(train_set)), tol=10 ** -3,n_jobs=-2)
                    model.fit(train_set.loc[:, train_set.columns != "Subgroup"], train_set.Subgroup)
                    # save number of features and prediction acuracy score
                    n_features = np.count_nonzero(model.coef_)
                    train_acc = accuracy(train_set, model)
                    test_acc = accuracy(test_set, model)
                    key = str(alpha) + "|" + str(l1r)
                    # get model predictions accuracies
                    not_her = list(train_predictions["predicted"] != "HER2+")
                    train_predictions.predicted[not_her] = model.predict(train_set.loc[:, train_set.columns != "Subgroup"])
                    tot_train_acc = sum(train_predictions.Subgroup == train_predictions.predicted)/float(len(train_predictions))
                    not_her = list(test_predictions["predicted"] != "HER2+")
                    test_predictions.predicted[not_her] = model.predict(test_set.loc[:, test_set.columns != "Subgroup"])
                    tot_test_acc = sum(test_predictions.Subgroup == test_predictions.predicted)/float(len(test_predictions))
                    # save to dict
                    if key in final_result.keys():
                        final_result[key]["rank"].append(rank)  #
                        final_result[key]["test_acc"].append(test_acc)
                        final_result[key]["train_acc"].append(train_acc)
                        final_result[key]["n_features"].append(n_features)
                        final_result[key]["tot_test_acc"].append(tot_test_acc)  #
                        final_result[key]["tot_train_acc"].append(tot_train_acc) #
                    else:
                        final_result[key] = {}
                        final_result[key]["rank"] = [rank]  #
                        final_result[key]["test_acc"] = [test_acc]
                        final_result[key]["train_acc"] = [train_acc]
                        final_result[key]["n_features"] = [n_features]
                        final_result[key]["tot_test_acc"] = [tot_test_acc] #
                        final_result[key]["tot_train_acc"] = [tot_train_acc] #
                        # written acc i.e. 5/7 ?

    # save dict on disc
    np.save("regularisation_results.npy",final_result)
    # Read dict to obtain optimal parameter
    read_dict = np.load("regularisation_results.npy").item()
    b = pd.DataFrame(read_dict)
    for i in b.index:
        for j in b.columns:
            b.loc[i,j] = np.mean(b.loc[i,j])
    b= b.transpose()
    placements = []
    for key in read_dict.keys():
        placements.append(len(read_dict[key]["rank"]))
    b["placements"] = placements
    b.sort_values("placements")
    q = b.iloc[-1].name
    print("alpha|l1_ratio:", q)
    mean_validation_acc = b.iloc[-1][2]
    print("average cv accuracy:", mean_validation_acc)
