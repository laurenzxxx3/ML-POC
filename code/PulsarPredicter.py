
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def read_and_split_data(csv):
	pulsar_data = pd.read_csv(csv)
	features = pulsar_data.drop(['target_class'], axis=1)
	labels = pulsar_data['target_class']
	return train_test_split(features, labels, test_size = 0.33, random_state = 42)

def decision_tree_model(train_features, train_labels, test_features, test_labels):
	tree = DecisionTreeClassifier()
	tree.fit(train_features, train_labels)
	tree_output = tree.predict(test_features)
	tree_accuracy = accuracy_score(test_labels, tree_output)
	tree_roc = roc_auc_score(test_labels, tree_output)
	return tree_output, tree_accuracy, tree_roc

def k_neighbors_model(train_features, train_labels, test_features, test_labels):
	knn = KNeighborsClassifier()
	knn.fit(train_features, train_labels)
	knn_output = knn.predict(test_features)
	knn_accuracy = accuracy_score(test_labels, knn_output)
	knn_roc = roc_auc_score(test_labels, knn_output)
	return knn_output, knn_accuracy, knn_roc

def random_forest_model(train_features, train_labels, test_features, test_labels)
	rf = RandomForestClassifier()
	rf.fit(train_features,train_labels)
	rf_output = rf.predict(test_features)
	rf_accuracy = accuracy_score(test_labels, rf_output)
	rf_roc = roc_auc_score(test_labels, rf_output)
	return rf_output, rf_accuracy, rf_roc


def main(csv):
	train_features, test_features, train_labels, test_labels = read_and_split_data(csv)
	tree_output, tree_accuracy, tree_roc = decision_tree_model(train_features, train_labels, test_features, test_labels)
	knn_output, knn_accuracy, knn_roc = k_neighbors_model(train_features, train_labels, test_features, test_labels)
	rf_output, rf_accuracy, rf_roc = random_forest_model(train_features, train_labels, test_features, test_labels)
	print ('tree_accuracy:',tree_accuracy)
	print ('tree ROC:',tree_roc)
	print ('knn_accuracy:',knn_accuracy)
	print ('knn ROC:',knn_roc)
	print ('rf accuracy',rf_accuracy)
	print ('rf AUC',rf_roc)

if __name__ == '__main__':
	print('Hi, I predict Pulsar Data!')
	if len(sys.argv) > 1:
		main(sys.argv[1])
	else:
		#TODO implement args with arg parse
		print('Provide path to csv')