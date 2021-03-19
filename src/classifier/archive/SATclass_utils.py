import numpy as np
import scipy
import sklearn
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from itertools import combinations
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import datasets, svm, metrics

mpl.rcParams.update({
        "font.size": 10.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 10.0,
        "text.usetex": False
})


def load(path):
    return {"verification" : pd.read_pickle("{}/all_verification.names.pkl".format(path)),
                   "agile" : pd.read_pickle("{}/all_agile.names.pkl".format(path)),
                  "crypto" : pd.read_pickle("{}/all_crypto.names.pkl".format(path)),
                 "crafted" : pd.read_pickle("{}/all_crafted.names.pkl".format(path)),
                  "random" : pd.read_pickle("{}/all_random.names.pkl".format(path))}

def collate(dictionary):
    # NOTE: We sample the minimum number of rows from each dataset.
    # TODO: Switch minrows off if you don't want to do category classification.
    minrows = np.min([dataframe.shape[0] for dataframe in dictionary.values()])
    dataframe = pd.concat([addhardness(addcategory(dataframe.sample(minrows), ID)) 
                           for (ID, dataframe) in enumerate(dictionary.values())])
    # Drop "rootInterEdges/rootInterVars" since it's computed incorrectly
    # dataframe = dataframe.drop(['rootInterEdges/rootInterVars'], axis=1) 
    return dataframe 

def addcategory(dataframe, ID):
    dataframe.insert(0, "Category", ID)
    return dataframe

def computehardness(solvingTime):
    return [np.median(solvingTime), np.inf]

def addhardness(dataframe):
    dataframe.insert(0, "Hardness", np.digitize(dataframe["solvingTime"].values, computehardness(dataframe["solvingTime"].values)))
    return dataframe

def extract(dataframe, label):
    labels   = dataframe[label].values
    features = dataframe.drop(["solvingTime", "Category", "Hardness", "SAT"], axis=1)
    return (labels, features.to_numpy(), features.columns)

def project(features, dimension):
    return PCA(n_components=dimension).fit_transform(features)

def classifier(dataframe, label):
    classifiers = [KNeighborsClassifier(3),
                   RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)]
    labels, features, columnheaders = extract(dataframe, label)
    scaledfeatures = StandardScaler().fit_transform(features) 
    # print("Balance: {}".format(np.unique(labels, return_counts=True%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def colour(labels):
    colours = {0 : "r", 1 : "b", 2 : "g", 3 : "c", 4 : "k", 5 : "y", 6 : "m", 7 : "tab:orange"}
    return list(map(lambda x : colours[x], labels))

def tSNE(features, labels):
    perplexity = [20, 30, 40, 50]
    scaledfeatures = StandardScaler().fit_transform(features) 
    projectedscaledfeatures = project(scaledfeatures, 50)  
    fig = plt.figure(figsize=(20,4))
    for (index, perp) in enumerate(perplexity):
        features_embedded = TSNE(n_components=2, perplexity=perp, n_iter=5000).fit_transform(projectedscaledfeatures)
        plt.subplot(1, 4, index+1)
        plt.scatter(features_embedded[:,0], features_embedded[:,1], color=colour(labels), alpha=0.4)
        plt.title(perp)

def compareMNIST(): 
    """ 
    Compare classifier results against MNIST
    """
    digits = datasets.load_digits()
    images_and_labels = list(zip(digits.images, digits.target))
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    for clf in classifiers:
        scores   = cross_val_score(clf, data, digits.target, cv=5)
        print(scores)
        print(str(clf) + " : Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def process(filename):
	dataframe = collate(load("../../datasets/{}".format(filename)))
	ALL   = dataframe 
	SAT   = dataframe[dataframe.SAT.eq(1)]
	UNSAT = dataframe[dataframe.SAT.eq(0)]
	UNKWN = dataframe[dataframe.SAT.eq(-1)]
	return ALL, SAT, UNSAT, UNKWN 

def RF(features, label, debug=0):
    scores = cross_val_score(KNeighborsClassifier(3), StandardScaler().fit_transform(features), label, cv=2)
    if debug == 1:
        print("\t balance ", np.unique(label, 	return_counts=True))
        print("\t scores ", scores)
        print("\t RF" + " : Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return (list(features.columns), scores.mean(), scores.std()*2)

def KNN(features, label, debug=0):
    scores = cross_val_score(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), StandardScaler().fit_transform(features), label, cv=2)
    if debug == 1:
        print("\t balance ", np.unique(label, 	return_counts=True))
        print("\t scores ", scores)
        print("\t KNN" + " : Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return (list(features.columns), scores.mean(), scores.std()*2)

def extractfeatures(dataframe):
    return dataframe.drop(["solvingTime", "Category", "Hardness", "SAT"], axis=1)

def extractlabels(dataframe):
    return dataframe.filter(["solvingTime", "Category", "Hardness", "SAT"], axis=1)
