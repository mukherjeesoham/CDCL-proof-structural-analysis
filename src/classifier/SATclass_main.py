#-----------------------------------------------------
# SAT classifier
# Soham 11/2020
# Here's the question we want to answer. Given several families SAT instances
# and their solving times on a specific CDCL solver, can we 
# - build a classifier that assigns a new SAT instance to one of the families? 
# - build a classifier that classifies a SAT instance (within a particular) family as easy or hard?
#-----------------------------------------------------

from SATclass_utils import collate, load, classifier, process, extractfeatures, extractlabels, KNN, RF  

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 
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
from sklearn.inspection import permutation_importance

def classify(filename):
    print("Using dataset {}".format(filename))
    localdata = collate(load("../../datasets/{}".format(filename)))
    # Look at only SAT instances
    # FIXME: Why are we getting different set of datapoints
    #        everytime we run the solver?
    localdata = localdata[localdata.SAT.eq(1)]

    print("----------------------------------")
    print("Classification results for Category")
    print("----------------------------------")
    classifier(localdata, "Category")
    print("----------------------------------")
    print("Classification results for Hardness")
    print("----------------------------------")
    classifier(localdata, "Hardness")

#-----------------------------------------------------
# Do a classification for all categories together
#-----------------------------------------------------
classify("pickle_31_01_2021/ALL")

#-----------------------------------------------------
# Do a classification for categories separately
#-----------------------------------------------------
ALL, SAT, UNSAT, UNKWN = process("pickle_31_01_2021/ALL")

# Run a classifier for category
# Check if the dataset is balanced or not. They grossly
# aren't. The best we can do at the moment is to use the whole dataset for category
# and run hardness classifier for SAT instances 
if (1):
	print("\nAre the datasets balanced for category?")
	print("ALL\t {}".format(np.unique(ALL.Category, 	return_counts=True)))
	print("SAT\t {}".format(np.unique(SAT.Category, 	return_counts=True)))
	print("UNSAT\t {}".format(np.unique(UNSAT.Category, return_counts=True)))
	
	print("\nAre the datasets balanced for Hardness?")
	print("ALL\t {}".format(np.unique(ALL.Hardness, 	return_counts=True)))
	print("SAT\t {}".format(np.unique(SAT.Hardness, 	return_counts=True)))
	print("UNSAT\t {}".format(np.unique(UNSAT.Hardness, return_counts=True)))
	
	print("\nAre the datasets balanced for SAT/UNSAT?")
	print("ALL\t {}".format(np.unique(ALL.SAT,  	return_counts=True)))
	print("SAT\t {}".format(np.unique(SAT.SAT,  	return_counts=True)))
	print("UNSAT\t {}".format(np.unique(UNSAT.SAT,  return_counts=True)))


F    = extractfeatures(ALL)
L    = extractlabels(ALL)
FSAT = extractfeatures(SAT)
LSAT = extractlabels(SAT)

if (1): 
    # Run a test classifier for category
    print("\nCategory")
    RF(F, L["Category"], 1)
    
    # Run a test classifier for hardness
    print("Hardness")
    RF(F, L["Hardness"], 1)
    
    # Run a test classifier for SAT/UNSAT
    print("SAT/UNSAT")
    RF(F, L["SAT"], 1)

if (1):
    # Feature importance using permutation importance
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1) 
    
    # Look at the confusion matrix for category
    cftrain, cftest, cltrain, cltest = train_test_split(F, L["Category"], random_state=0)
    clf.fit(cftrain, cltrain)
    print("\n--------------------------------")
    print("Category")
    print("--------------------------------")
    print("RF train accuracy: %0.3f" % clf.score(cftrain, cltrain))
    print("RF test accuracy: %0.3f" % clf.score(cftest, cltest))
    # plot_confusion_matrix(clf, cftest, cltest)  
    # plt.show()  
    
    # Look at permutation importance
    result = permutation_importance(clf, cftest, cltest, n_repeats=10, random_state=42, n_jobs=2)
    sorted_idx  = result.importances_mean.argsort()
    topfeatures = F.columns[sorted_idx].to_numpy()
    print(topfeatures[:5])
    
    # Look at the confusion matrix for hardness
    hftrain, hftest, hltrain, hltest = train_test_split(F, L["Hardness"], random_state=0)
    clf.fit(hftrain, hltrain)
    print("\n--------------------------------")
    print("Hardness")
    print("--------------------------------")
    print("RF train accuracy: %0.3f" % clf.score(hftrain, hltrain))
    print("RF test accuracy: %0.3f" % clf.score(hftest, hltest))
    # plot_confusion_matrix(clf, hftest, hltest)  
    # plt.show()
    
    result = permutation_importance(clf, hftest, hltest, n_repeats=10, random_state=42, n_jobs=2)
    sorted_idx  = result.importances_mean.argsort()
    topfeatures = F.columns[sorted_idx].to_numpy()
    print(topfeatures[:5])

if(1):
    print("\nStarting computations for drop-feature importance.\n")
    # Find out the top 5 features that predict Category using drop feature importance
    QC = Parallel(n_jobs=4, verbose=1, prefer="threads")(delayed(RF)(F.filter(F5), L["Category"]) for F5 in list(combinations(F, 5)))
    QC.sort(key = lambda l: l[1])
    print("\n--------------------------------")
    print("Top 5: category")
    print("--------------------------------")
    print("\t", QC[-3])
    print("\t", QC[-2])
    print("\t", QC[-1])
    
    # Find out the top 5 features that predict Hardness using drop feature importance
    QH = Parallel(n_jobs=4, verbose=1, prefer="threads")(delayed(RF)(F.filter(F5), L["Hardness"]) for F5 in list(combinations(F, 5)))
    QH.sort(key = lambda l: l[1])
    print("\n--------------------------------")
    print("Top 5: Hardness")
    print("--------------------------------")
    print("\t", QH[-3])
    print("\t", QH[-2])
    print("\t", QH[-1])
