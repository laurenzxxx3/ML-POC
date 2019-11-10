import sys
import pandas as pd
import numpy as np


def normalize_data(features, labels):
	for col in range(features.shape[1]):
		max_x = features.iloc[:,col].max()
		min_x = features.iloc[:,col].min()
		for row in range(features.shape[0]):
			features.iloc[[row],[col]] = (float(features.iat[row,col])-min_x)/(max_x-min_x)
	pulsar_data_normalized = pd.concat([features, labels], axis=1)
	pulsar_data_normalized.to_csv('../pulsar_data_normalized.csv')
	print ('written data to file pulsar_data_normalized.csv')


def main(csv):
	pulsar_data = pd.read_csv(csv)
	labels = pulsar_data['target_class']
	features = pulsar_data.drop(['target_class'],axis=1)
	normalize_data(features, labels)

if __name__ == '__main__':
	if len(sys.argv) > 1:
		main(sys.argv[1])
	else:
		print('Provide path to csv')