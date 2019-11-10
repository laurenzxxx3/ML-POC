#!/usr/bin/python3

# Imports
import argparse
import numpy as np
import pandas as pd
import reg_model
import rfe_model
import uni_model

# Functions
def preprocessing(features, labels):
    features = pd.read_csv(features)
    labels = pd.read_csv(labels)

    # Split instances from biological data
    chr_info = features.iloc[:,0:3]
    arrays = features.iloc[:,4:]
    arrays = arrays.T
    arrays['Subgroup'] = list(labels['Subgroup'])
    use_data = arrays.T
    # Indicate first split feature
    gate = use_data.iloc[2184]
    # Create two empty dataframes
    her2_df = pd.DataFrame()
    not_her2_df = pd.DataFrame()
    # Loop through instances
    for i in range(len(use_data.columns)):
        # Check value of first-split-feature per row
        if use_data.iloc[2184,i] >= 2:
            # If value == 2, instance is HER2+
            name = use_data.columns[i]
            her2_df[name] = use_data[name]
        else:
            # Else, instance is not HER2+
            name = use_data.columns[i]
            not_her2_df[name] = use_data[name]
    # Transpose both dataframes for easy use
    her2_df = her2_df.T
    not_her2_df =not_her2_df.T

    # Return split data
    return(her2_df, not_her2_df)


def parser():
    """Retrieves the arguments from the command line.
    """

    # create argument parser
    parser = argparse.ArgumentParser(description='A program to run the validation script')
    # initiate data arguments
    parser.add_argument('-f', metavar= 'features', dest= 'features', help= '[-f] to select features for predictions')
    parser.add_argument('-l', metavar= 'labels', dest= 'labels', help= '[-l] to select labels for training')
    # initiate function arguments
    parser.add_argument('-m', metavar= 'model', dest= 'model', help= '[-m] to select: ELA, RFE or UNI for training and validation')
    # place arguments in variable
    arguments = parser.parse_args()
    # return chosen function and data
    return(arguments)

def main():
    """
    A function to retrieve the data and run the other functions
    """

    # Get arguments
    args = parser()
    features = args.features
    labels = args.labels
    model = args.model
    # Split her2_samples from the not her2 samples
    her2_samples, not_her2_samples = preprocessing(features, labels)
    # Check which model is chosen for training and run said model
    if model == 'ELA':
        reg_model.elasticnet(not_her2_samples)
    elif model == 'RFE':
        rfe_model.rfe(not_her2_samples)
    elif model == 'UNI':
        uni_model.uni(not_her2_samples)
    else:
        print('no model chosen, choose between ELA, RFE or UNI')


if __name__ == '__main__':
    main()

# Last line
