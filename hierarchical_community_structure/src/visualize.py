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

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

mpl.rcParams.update({
        "font.size": 18.0,
        "axes.titlesize": 12.0,
        "axes.labelsize": 12.0,
        "xtick.labelsize": 12.0,
        "ytick.labelsize":12.0,
        "legend.fontsize": 12.0,
        "text.usetex": True
})

def colour(labels):
    colours = {0 : "r", 1 : "b", 2 : "g", 3 : "c", 4 : "k"}
    return list(map(lambda x : colours[x], labels))

def tSNE(dataframe, label):
    perplexity = [20, 30, 40, 50]
    labels, features = extract(dataframe, label)
    scaledfeatures = StandardScaler().fit_transform(features) 
    scores   = cross_val_score(clf, scaledfeatures, labels, cv=10)
    fig = plt.figure(figsize=(20,4))
    for (index, perp) in enumerate(perplexity):
        features_embedded = TSNE(n_components=2, perplexity=perp, n_iter=1000).fit_transform(features)
        plt.subplot(1, 4, index+1)
        plt.scatter(features_embedded[:,0], features_embedded[:,1], color=colour(labels))
        plt.title(perp)
    plt.show()

