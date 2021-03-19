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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline 
from sklearn import preprocessing

def load(path):
    return {"verification" : pd.read_pickle("{}/all_verification.names.pkl".format(path)),
                   "agile" : pd.read_pickle("{}/all_agile.names.pkl".format(path)),
                  "crypto" : pd.read_pickle("{}/all_crypto.names.pkl".format(path)),
                  # "random" : pd.read_pickle("{}/all_random.names.pkl".format(path)),
                 "crafted" : pd.read_pickle("{}/all_crafted.names.pkl".format(path))}

def extractfeatures(dataframe):
    return dataframe.drop(["solvingTime", "SAT", "Category"], axis=1)

def extractcolumns(dataframe):
    return dataframe.drop(["solvingTime", "SAT", "Category"], axis=1).columns

def extractlabels(dataframe):
    return dataframe.filter(["Category"], axis=1)

def RandomForest(features, labels, columns):
    reg = RandomForestClassifier()   # NEW: If you don't know what hyperparameters are good, don't set any of them.
    print(np.unique(labels.to_numpy(), return_counts=True))
    rftrain, rftest, rltrain, rltest = train_test_split(features, labels.to_numpy().ravel())
    reg.fit(rftrain, rltrain)
    trainscore = reg.score(rftrain, rltrain)
    testscore = reg.score(rftest, rltest)
    rankedfeatures = sorted(zip(map(lambda x: round(x, 4), reg.feature_importances_), columns), reverse=True) # Compute feature importances

    if (0):
        print("\t RF classifier train accuracy: %0.3f" %trainscore)
        print("\t RF classifier test accuracy: %0.3f" %testscore)
        print(ranks[:10])
    return (trainscore, testscore, rankedfeatures[:4]) 


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
    print(features.columns)
    features = scaler.fit_transform(features)
    return features, label

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

def prune(df, counts):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # TODO: Use the count to drop the correlated feature.  
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    df.drop(to_drop, axis=1, inplace=True)
    return df.columns

def iterate(dictionary):
    for (key, dataframe) in dictionary.items():
        if key == 'all':
            print("----------------------------------")
            print(key)
            print("----------------------------------")
            describe(dataframe)
            preprocess(dataframe)
            describe(dataframe)
            dcopy = dataframe.copy()
            features, label = transform(extractfeatures(dataframe), extractlabels(dataframe))
            columns = extractcolumns(dataframe)     
            print(features.shape)
            unique, counts  = collect(10, features, label, columns) 
            print(unique)
            prunedcolumns = prune(extractfeatures(dataframe)[unique], counts)
            print("\t Top uncorrelated features: ", list(prunedcolumns))

def addcategory(dataframe, ID):
    dataframe.insert(0, "Category", ID)
    return dataframe

def collate(dictionary):
    minrows = np.min([dataframe.shape[0] for dataframe in dictionary.values()])
    dataframe = pd.concat([addcategory(dataframe.sample(minrows), ID) for (ID, dataframe) in enumerate(dictionary.values())])
    return dataframe 

# Load and combine data, and run the analyses
dictionary = load("../../datasets/pickle_13_02_2021/ALL")
# Join all the dataframes
dictionary["all"] = collate(dictionary)
iterate(dictionary)

