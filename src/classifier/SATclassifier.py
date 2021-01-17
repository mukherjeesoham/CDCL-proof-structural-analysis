import numpy as np
import scipy
import sklearn
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
        "xtick.labelsize": 10.0,
        "ytick.labelsize":10.0,
        "legend.fontsize": 10.0,
        "text.usetex": True
})


def load(path):
    return {"verification" : pd.read_pickle("{}/jonathan_all_verification.names.pkl".format(path)),
                   "agile" : pd.read_pickle("{}/jonathan_all_agile.names.pkl".format(path)),
                  "crypto" : pd.read_pickle("{}/jonathan_all_crypto.names.pkl".format(path)),
                 "crafted" : pd.read_pickle("{}/jonathan_all_crafted.names.pkl".format(path)),
                  "random" : pd.read_pickle("{}/jonathan_all_random.names.pkl".format(path))}

def collate(dictionary, hardness):
    # TODO: Sample the minimum number of rows from each dataset.
    minrows = np.min([dataframe.shape[0] for dataframe in dictionary.values()])
    return pd.concat([addhardness(addcategory(dataframe.sample(minrows), ID), list(hardness.values())) 
                        for (ID, dataframe) in enumerate(dictionary.values())])

def addcategory(dataframe, ID):
    dataframe.insert(0, "Category", ID)
    return dataframe

def addhardness(dataframe, upperbounds):
    dataframe.insert(0, "Hardness", np.digitize(dataframe["solvingTime"].values, upperbounds))
    return dataframe

def extract(dataframe, label):
    labels   = dataframe[label].values
    features = dataframe.drop(["solvingTime", "Category", "Hardness"], axis=1)
    # TODO: Add filter and work with a smaller subset features
    features = features.filter(['numVars', 'numClauses', 'modularity', 'mergeabilitynorm1', 'leafCommunitySize', 'depth', 'leafInterVars', 'leafInterEdges', 'k'])
    # TODO: What happens if we only use one or two features?
    # features = features.filter(['leafInterVars', 'leafInterEdges'])
    return (labels, features.to_numpy(), features.columns)

def project(features, dimension):
    return PCA(n_components=dimension).fit_transform(features)

def classifier(dataframe, label):
    # List of classifiers we're using
    classifiers = [
        KNeighborsClassifier(3),
        # SVC(kernel="linear", C=0.025), 
        # SVC(gamma=2, C=1),
        # DecisionTreeClassifier(max_depth=5), 
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        # MLPClassifier(alpha=1, max_iter=1000),
        # AdaBoostClassifier(), 
        # GaussianNB()]
        ]

    labels, features, columnheaders = extract(dataframe, label)
    scaledfeatures = StandardScaler().fit_transform(features) 
    # print(np.unique(labels, return_counts=True))

    for clf in classifiers:
        scores = cross_val_score(clf, scaledfeatures, labels, cv=5)
        print("\t", scores)
        print(str(clf) + " : Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

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

# Compare classifier results against MNIST
def compareMNIST(): 
    digits = datasets.load_digits()
    images_and_labels = list(zip(digits.images, digits.target))
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    for clf in classifiers:
        scores   = cross_val_score(clf, data, digits.target, cv=5)
        print(scores)
        print(str(clf) + " : Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
