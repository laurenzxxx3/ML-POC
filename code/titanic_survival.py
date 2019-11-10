import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

def read_data(csv1,csv2,csv3):
	# read
	train_data = pd.read_csv(str(csv1))
	test_features = pd.read_csv(str(csv2))
	test_labels = pd.read_csv(str(csv3))
	# shuffle
	#data = data.sample(frac=1)
	# change features to numerical
	labelencoder = LabelEncoder()
	train_data.Sex = labelencoder.fit_transform(train_data.Sex)
	train_data.Embarked = labelencoder.fit_transform(train_data.Embarked)
	test_features.Sex = labelencoder.fit_transform(test_features.Sex)
	test_features.Embarked = labelencoder.fit_transform(test_features.Embarked)	
	# seperate survivors
	# seperate features
	train_features = train_data[['Sex','Pclass','Embarked','Child']]
	train_labels = train_data['Survived']
	test_features = test_features[['Sex','Pclass','Embarked','Child']]
	test_labels = test_labels['Survived']
	#features = data.drop(['PassengerId','Survived','Name','Age','SibSp','Parch','Ticket','Cabin','Fare'])
	# split train and test set
	#train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.33, random_state = 42)
	return train_features, test_features, train_labels, test_labels

def model(train_features,train_labels,test_features,test_labels):
	tree = DecisionTreeClassifier()
	knn = KNeighborsClassifier()
	#tree_scores = cross_val_score(tree, train_features, train_labels, cv=3, scoring = 'roc_auc')
	#print (tree_scores)
	tree.fit(train_features, train_labels)
	knn.fit(train_features, train_labels)
	# predict
	tree_output = tree.predict(test_features)
	tree_accuracy = accuracy_score(test_labels, tree_output)
	tree_roc = roc_auc_score(test_labels, tree_output)
	knn_output = knn.predict(test_features)
	knn_accuracy = accuracy_score(test_labels, knn_output)
	knn_roc = roc_auc_score(test_labels, knn_output)
	knn_df = pd.DataFrame()
	knn_df["Prediction"] = knn_output
	knn_df["Actual"] = test_labels
	knn_df.to_csv('knn_output.csv')
	tree_df = pd.DataFrame()
	tree_df["Prediction"] = tree_output
	tree_df["Actual"] = test_labels
	tree_df.to_csv('tree_output.csv')
	return tree_output, tree_accuracy, tree_roc, knn_output, knn_accuracy, knn_roc

def main():
	csv1 = sys.argv[1]
	csv2 = sys.argv[2]
	csv3 = sys.argv[3]
	train_features, test_features, train_labels, test_labels = read_data(csv1,csv2,csv3)
	tree_output, tree_accuracy, tree_roc, knn_output, knn_accuracy, knn_roc = model(train_features,train_labels,test_features,test_labels)
	#print ('tree predictions:', tree_output)
	print ('tree_accuracy:',tree_accuracy)
	print ('tree ROC:',tree_roc)
	#print ('knn predictions:', knn_output)
	print ('knn_accuracy:',knn_accuracy)
	print ('knn ROC:',knn_roc)	


if __name__ == '__main__':
    main()
