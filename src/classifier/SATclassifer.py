#-----------------------------------------------------
# SATClassifier.py
# Soham 11/2020
# Here's the question we want to answer. Given several families SAT instances
# and their solving times on a specific CDCL solver, can we 
# - build a classifier that assigns a new SAT instance to one of the families? 
# - build a classifier that classifies a SAT instance (within a particular) family as easy or hard?
#-----------------------------------------------------

import numpy as np
import scipy
import sklearn
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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

mpl.rcParams.update({
        "font.size": 18.0,
        "axes.titlesize": 12.0,
        "axes.labelsize": 12.0,
        "xtick.labelsize": 12.0,
        "ytick.labelsize":12.0,
        "legend.fontsize": 12.0,
        "text.usetex": True
})

def load(path):
    return {"verification" : pd.read_pickle("{}/all_verification.names.pkl".format(path)),
                   "agile" : pd.read_pickle("{}/all_agile.names.pkl".format(path)),
                  "crypto" : pd.read_pickle("{}/all_crypto.names.pkl".format(path)),
                 "crafted" : pd.read_pickle("{}/all_crafted.names.pkl".format(path)),
                  "random" : pd.read_pickle("{}/all_random.names.pkl".format(path))}

def collate(dictionary, hardness):
    return pd.concat([addhardness(addcategory(dataframe, ID), list(hardness.values())) 
                        for (ID, dataframe) in enumerate(dictionary.values())])

def addcategory(dataframe, ID):
    dataframe.insert(0, "Category", ID)
    return dataframe

def addhardness(dataframe, upperbounds):
    dataframe.insert(0, "Hardness", np.digitize(dataframe["solvingTime"].values, upperbounds))
    return dataframe

def extract(dataframe, label):
    labels   = dataframe[label].values
    features = dataframe.drop(["solvingTime", "Category", "Hardness"], axis=1).to_numpy()
    return (labels, features)

def project(features, dimension):
    return PCA(n_components=dimension).fit_transform(features)

def classifier(dataframe, label):
    for clf in classifiers:
        labels, features = extract(dataframe, label)
        scaledfeatures = StandardScaler().fit_transform(features) 
        scores   = cross_val_score(clf, scaledfeatures, labels, cv=10)
        print(str(clf) + " : Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def colour(labels):
    colours = {0 : "r", 1 : "b", 2 : "g", 3 : "c", 4 : "k"}
    return list(map(lambda x : colours[x], labels))

def tSNE(dataframe, label):
    perplexity = [20, 30, 40, 50]
    # Modifications to test individual classes/categories
    # dataframe = dataframe[dataframe["Category"] == 4]

    labels, features = extract(dataframe, label)
    scaledfeatures = StandardScaler().fit_transform(features) 
    projectedscaledfeatures = project(scaledfeatures, 50)  
    fig = plt.figure(figsize=(20,4))

    for (index, perp) in enumerate(perplexity):
        features_embedded = TSNE(n_components=2, perplexity=perp, n_iter=5000).fit_transform(projectedscaledfeatures)
        plt.subplot(1, 4, index+1)
        plt.scatter(features_embedded[:,0], features_embedded[:,1], color=colour(labels))
        plt.title(perp)
    plt.show()

#-----------------------------------------------------
# Run on datasets
#-----------------------------------------------------

hardness = {
    "easy" : np.log2(60),
    "hard" : np.inf
}

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),  # SVC are sensitive. Type of kernel is important.
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5), # Can overfit
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), # Good at avoiding overfitting > estimators > robust
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(), # Avoids overfitting; sequential instead of parallel. Change the number of iterations.
    GaussianNB()
]

# Compare classifier results against MNIST
from sklearn import datasets, svm, metrics
digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
for clf in classifiers:
    scores   = cross_val_score(clf, data, digits.target, cv=5)
    print(str(clf) + " : Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

data = collate(load("../../datasets/pickle"), hardness)
classifier(data, "Category")
print("----------------------------------")
classifier(data, "Hardness")

