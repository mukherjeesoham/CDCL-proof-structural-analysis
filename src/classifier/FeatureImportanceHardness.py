#-------------------------------------------------------------------------
# Compute feature importance and then look for good/bad parameters
# Soham 02/2021
#-------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os

from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline 
from sklearn import preprocessing

def load(path):
    return {"verification" : pd.read_pickle("{}/all_verification.names.pkl".format(path)),
                   "agile" : pd.read_pickle("{}/all_agile.names.pkl".format(path)),
                  "crypto" : pd.read_pickle("{}/all_crypto.names.pkl".format(path)),
                 "crafted" : pd.read_pickle("{}/all_crafted.names.pkl".format(path))}

def join(dictionary):
    return pd.concat([dataframe for (key, dataframe) in dictionary.items()])

def extractfeatures(dataframe):
    return dataframe.drop(["solvingTime", "SAT"], axis=1)

def extractcolumns(dataframe):
    return dataframe.drop(["solvingTime", "SAT"], axis=1).columns

def extractlabels(dataframe):
    return dataframe.filter(["solvingTime"], axis=1)

def describe(dataframe):
    print("\t All                 : {}".format(dataframe.shape[0]))
    print("\t SolvingTime < 1e-15 : {}".format(dataframe[dataframe['solvingTime'] < 1e-15]['solvingTime'].size))
    print("\t SolvingTime > 4949  : {}".format(dataframe[dataframe['solvingTime'] > 4949]['solvingTime'].size))
    print("\t Indeterminate       : {}".format(dataframe[dataframe.SAT == -1].shape[0]))
    
def preprocess(dataframe):
    dataframe.drop(dataframe[dataframe.SAT == -1].index, inplace=True)              # Remove Indeterminate
    dataframe.drop(dataframe[dataframe.solvingTime == 0.0].index, inplace=True)     # NEW: Remove instances with solving time of zero.
    return dataframe

def transform(features, label):
    scaler = preprocessing.StandardScaler()
    features = scaler.fit_transform(features)
    label = np.log10(label)
    return features, label

def RandomForest(features, labels, columns):
    features, labels = transform(features, labels)
    reg = RandomForestRegressor()   # NEW: If you don't know what hyperparameters are good, don't set any of them.
    rftrain, rftest, rltrain, rltest = train_test_split(features, labels.to_numpy().ravel())
    reg.fit(rftrain, rltrain)
    trainscore = reg.score(rftrain, rltrain)
    testscore = reg.score(rftest, rltest)
    rankedfeatures = sorted(zip(map(lambda x: round(x, 4), reg.feature_importances_), columns), reverse=True) # Compute feature importances

    if (0):
        print("\t RF regressor train accuracy: %0.3f" %trainscore)
        print("\t RF regressor test accuracy: %0.3f" %testscore)
        print(ranks[:10])
    return (trainscore, testscore, rankedfeatures[:4]) 

def collect(iterations, features, label, columns):
        trainlist = []
        testlist  = []
        top5list  = [] 
        for i in range(iterations):
            trainscore, testscore, rankedfeatures = RandomForest(features, label, columns)
            trainlist.append(trainscore)
            testlist.append(testscore)
            top5list.append(rankedfeatures)
        print("\t train accuracy = {} +\- {}".format(np.mean(trainlist), np.std(np.array(trainlist))))
        print("\t test accuracy = {} +\- {}".format(np.mean(testlist), np.std(np.array(testlist))))
        merged = list(itertools.chain.from_iterable(top5list))
        merged = list(sum(merged, ())) 
        merged = np.array(merged).reshape(-1,2)
        unique, counts = np.unique(merged[:,1], return_counts=True)
        return unique, counts

def prune(dataframe, params, threshold):
    corrmatrix = dataframe.corr().abs() 
    for param in params:
        for column in corrmatrix.columns:
            if column != param:
                if (corrmatrix[param][column] > threshold):
                    if column not in params: 
                        dataframe.drop(column, axis=1, inplace=True, errors='ignore')
    return dataframe

def trim(dataframe, counts, threshold):
    corrmatrix = dataframe.corr().abs() 
    for ix, px in enumerate(params):
        for iy, py in enumerate(params):
                if (corrmatrix[px][py] > threshold):
                    if counts[ix] > counts[iy]:
                        dataframe.drop(px, axis=1, inplace=True, errors='ignore')
                    elif counts[iy] > counts[ix]:
                        dataframe.drop(py, axis=1, inplace=True, errors='ignore')
    return dataframe

def iterate(dictionary):
    for (key, dataframe) in dictionary.items():
        # if key == "all": 
        print("----------------------------------")
        print(key)
        print("----------------------------------")
        describe(dataframe)
        preprocess(dataframe)  
        describe(dataframe)
        features, label, columns = extractfeatures(dataframe), extractlabels(dataframe), extractcolumns(dataframe)
        # prune(features, params, 0.80) # NEW: Prune all parameters before running RandomForest
        unique, counts  = collect(10, features, label, columns) 
        print(list(sorted(zip(counts, unique), reverse=True)))


# Load and combine data, and run the analyses
dictionary = load("../../datasets/pickle_13_02_2021/ALL")
dictionary["all"] = join(dictionary)

# Prune all the highly correlated parameters, making sure the 
# the parameters in params are kept.
params = ['rootInterEdges/rootCommunitySize',
        'rootInterVars/rootCommunitySize', 'lvl2InterEdges/lvl2CommunitySize',
        'lvl2InterVars/lvl2CommunitySize', 'lvl3InterEdges/lvl3CommunitySize',
        'lvl3InterVars/lvl3CommunitySize', 'rootModularity', 'lvl2Modularity',
        'lvl3Modularity']  

iterate(dictionary)

