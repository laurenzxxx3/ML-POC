import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


def feature_correlation(dataset):
	# plot Pearson Correlation
	plt.figure(figsize=(10,8))
	cor = dataset.corr()
	sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
	plt.show()

def feature_importances(dataset):
	features = dataset.drop(['target_class'],axis=1)
	labels = np.array(dataset['target_class'])
	# extract feature coefficients of training contribution
	reg = LassoCV()
	reg.fit(features, labels)
	coef = pd.Series(reg.coef_, index = features.columns)
	imp_coef = coef.sort_values()
	matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
	imp_coef.plot(kind = "barh")
	plt.title("Feature importance using Lasso Model")
	plt.show()


def main(csv):
	pulsar_data = pd.read_csv(csv)
	feature_correlation(pulsar_data)
	feature_importances(pulsar_data)

if __name__ == '__main__':
	if len(sys.argv) > 1:
		main(sys.argv[1])
	else:
		print('Provide path to csv')