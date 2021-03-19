#-------------------------------------------------------------------------
# Ask why the classifier is not working as well as we want. Is the problem
# difficult or are we choosing the wrong metrics?
# Soham 02/2021

# - Do our computations work only for some subcategories? Start with tSNE and
# then look at classifier results for hardness.  
# - Use a Random Forest Regressor to avoid the issue of bucketing. 
# - Use log of solvingTime. Add a small number to make sure log doesn't
# complain. 
# - Keep in mind about the artificial cutoff at 5000 seconds and for the
# samples with zero seconds as solving time. 
# - Use different buckets for classifiers. Further use a stratified split to
# avoid the issue of unbalanced datasets.  
#       > Using buckets for the classifier would be misleading because of
#         how unbalanced our dataset is.    
# - Choose features wisely, minimize correlations between features.
#       > Use the inbuilt feature selection options from sklearn
#-------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline 
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LassoCV # WTF is this?
from sklearn.decomposition import PCA
from sklearn import preprocessing

def load(path):
    return {"verification" : pd.read_pickle("{}/all_verification.names.pkl".format(path)),
                   "agile" : pd.read_pickle("{}/all_agile.names.pkl".format(path)),
                  "crypto" : pd.read_pickle("{}/all_crypto.names.pkl".format(path)),
                 "crafted" : pd.read_pickle("{}/all_crafted.names.pkl".format(path))}

def extractfeatures(dataframe):
    return dataframe.drop(["solvingTime", "SAT"], axis=1)

def extractlabels(dataframe):
    return dataframe.filter(["solvingTime"], axis=1)

def balanced_test_train_split(features, labels):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    for (train_index, test_index) in  sss.split(features, labels):
        return (features[train_index], features[test_index], labels[train_index], labels[test_index])

def tSNE(features, solvingTime):
    features_em = TSNE(n_components=2, perplexity=30, n_iter=2000).fit_transform(StandardScaler().fit_transform(features))
    plt.scatter(features_em[:,0], features_em[:,1], c=solvingTime.to_numpy(), alpha=0.4)
    plt.colorbar()
    plt.show()
    
def regression(features, labels, category):
    rftrain, rftest, rltrain, rltest = train_test_split(features, labels.to_numpy().ravel(), random_state=42)
    regr = Pipeline([('Scaler', preprocessing.StandardScaler()), 
                     # ('reduce_dim', PCA(n_components=15)),
                     # ('feature_selection',  SequentialFeatureSelector(LassoCV(), n_features_to_select=10, direction='backward', cv=3, n_jobs=-1)), 
                     # ('Regressor', RandomForestRegressor(max_depth=3,  n_estimators=3, max_features="auto", n_jobs=-1, random_state=42))])
                     ('Regressor', RandomForestRegressor(random_state=42))])
    regr.fit(rftrain, rltrain)
    print("\t RF regressor train accuracy: %0.3f" % regr.score(rftrain, rltrain))
    print("\t RF regressor test accuracy:  %0.3f" % regr.score(rftest, rltest))
       
    if (1):
        plt.scatter(rltest, regr.predict(rftest), s=6)
        plt.plot(rltest, rltest, "k--", linewidth=0.5)
        plt.xlabel("True solvingTime")
        plt.ylabel("Predicted solvingTime")
        plt.grid(True)
        plt.title("$R^2 = {}$".format(regr.score(rftest, rltest)))
        plt.savefig("{}.pdf".format(category))
        plt.close()
    return regr

def pca(n, features, labels): 
    features = preprocessing.StandardScaler().fit_transform(features)
    pca = PCA(n_components=n)
    pca.fit(features)
    return (np.sum(pca.explained_variance_ratio_))

def RandomForest(features, labelas, category):
    reg = RandomForestRegressor(max_depth=5,  n_estimators=10, max_features=1)
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1) 
    scaler = preprocessing.StandardScaler()
    tseconds = 100
    
    features = scaler.fit_transform(features)
    labels   = labels.to_numpy().ravel()
    blabels  = np.digitize(labels, [np.log10(tseconds)])
    
    rftrain, rftest, rltrain, rltest = train_test_split(features, labels)
    cftrain, cftest, cltrain, cltest = balanced_test_train_split(features, blabels)

    # NOTE: We get really high classifier accuracies. While this looks promising 
    # a priori, I strongly suspect the issue is with how unbalanced our dataset is.
    # For example, if you choose the median for classifying, you get much worse results. 
    # However, choosing the median is not a great way to test the algorihtm since there 
    # might exist no separation, and it won't be a fair comparison. 
    # Concerning the RandomForest regressor, how to figure out what's good enough? Compare
    # with the SATZilla paper Vijay sent.
    
    # print(np.unique(blabels, return_counts=True))
    # print(np.unique(cltest, return_counts=True))
    # print(np.unique(cltrain, return_counts=True))

    reg.fit(rftrain, rltrain)
    clf.fit(cftrain, cltrain)   
    
    print("\t RF regressor train accuracy: %0.3f" % reg.score(rftrain, rltrain))
    print("\t RF regressor test accuracy: %0.3f" % reg.score(rftest, rltest))
    print("\t RF classifier train accuracy: %0.3f" % clf.score(cftrain, cltrain))
    print("\t RF classifier test accuracy: %0.3f\n" % clf.score(cftest, cltest))
       
    if (1):
        fig = plt.figure()
        plt.scatter(rltest, reg.predict(rftest), s=6)
        plt.plot(rltest, rltest, "k--", linewidth=0.5)
        plt.xlabel("True solvingTime")
        plt.ylabel("Predicted solvingTime")
        plt.grid(True)
        plt.show()
    
    return reg, clf

def describe(dataframe):
    print("\t All                 : {}".format(dataframe.shape[0]))
    print("\t SolvingTime < 1e-15 : {}".format(dataframe[dataframe['solvingTime'] < 1e-15]['solvingTime'].size))
    print("\t SolvingTime > 4949  : {}".format(dataframe[dataframe['solvingTime'] > 4949]['solvingTime'].size))
    print("\t Indeterminate       : {}".format(dataframe[dataframe.SAT == -1].shape[0]))
    
def preprocess(dataframe):
    dataframe.drop(dataframe[dataframe.SAT == -1].index, inplace=True)
    dataframe.drop(dataframe[dataframe.solvingTime == 0.0].index, inplace=True)     # NEW: Remove instances with solving time of zero.
    dataframe.solvingTime = np.log10(dataframe.solvingTime)
    return dataframe

def iterate():
    from sklearn.preprocessing import MinMaxScaler
    for (key, dataframe) in load("../../datasets/pickle_13_02_2021/ALL").items():
        print(key)
        preprocess(dataframe)
        print("\t n components = {} Variance Ratio = {}".format(10, pca(10, extractfeatures(dataframe), extractlabels(dataframe))))
        regression(extractfeatures(dataframe), extractlabels(dataframe), key)
       
def combined(): 
    return None

def splitSATUNSAT():
    return None

iterate()

# TODO: 
# [0] Compute a baseline accuracy.
# [1] Do not do PCA/ICA. Focus on interpretability. To do this, compute feature
# importances using a RandomForest regressor to compute feature importances.
# [2] Choose only those features that sum upto 95% of the total importance. 
# [3] Now use these features to train the RandomForest regressor with the same parameters as [0].
# [4] Now do a random search over the parameters first. Find the range of parameters that look promising.
# [5] Then do a grid search for the best set of parameters.
# [6] Remove outliers? 
