#-----------------------------------------------------
# Run sets of classifiers (KNN and RandomForest) 
# on the new dataset given by Ian on Jan 16 2020. 
#-----------------------------------------------------

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from joblib import Parallel, delayed 
from SATclassifier import load, collate 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# TODO: 	
# [1] Run classifiers separately on SAT and UNSAT instances. 
# [2] Find top 5 features that's predictive of runtime. 
# [3] Find top 5 features that's predictive of category. 
# [4] Find top 5 features that's predictive of SAT/UNSAT. [Crazy; I know] 

hardness = {
    # The distribution is bi-modal with a valley 
    # starting at around 2000 seconds. Therefore, 
    # it doesn't make sense to have more than two
    # bins.
    "easy"      : np.log2(2264), 
    "hard"      : np.log2(np.inf)
}

features  = ["numVars", "numClauses", "cvr", "rootInterVars",
    		 "rootInterEdges", "leafCommunity", "leafModularity", "rootMergeability1norm1",
	 		 "rootMergeability1norm2", "rootMergeability2norm1",
	 		 "rootMergeability2norm2","rootModularity", "depth", "numLeaves", "dvMean",
	 		 "dvVariance","degreeMean", "degreeVariance"]
labels    = ["SAT", "Category", "Hardness", "SATHardness"]
regressor = ["solvingTime"]
debug     = 0

def process(filename):
	dataframe = collate(load("../../datasets/{}".format(filename)), hardness)
	ALL   = dataframe 
	# FIXME: What is UNKWN?
	SAT   = dataframe[dataframe.SAT.eq(0)]
	UNSAT = dataframe[dataframe.SAT.eq(-1)]
	UNKWN = dataframe[dataframe.SAT.eq(1)]
	return ALL, SAT, UNSAT, UNKWN 

def RF(features, label):
	scores = cross_val_score(KNeighborsClassifier(3), StandardScaler().fit_transform(features), label, cv=5)
	# print("\t balance ", np.unique(label, 	return_counts=True))
	# print("\t scores ", scores)
	# print("\t RF" + " : Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	return (list(features.columns), scores.mean(), scores.std()*2)

def KNN(features, label):
	scores = cross_val_score(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), StandardScaler().fit_transform(features), label, cv=5)
	# print("\t balance ", np.unique(label, 	return_counts=True))
	# print("\t scores ", scores)
	# print("\t KNN" + " : Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	return (list(features.columns), scores.mean(), scores.std()*2)

def extractfeatures(dataframe):
	return dataframe.filter(features)

def extractlabels(dataframe):
	return dataframe.filter(labels)

ALL, SAT, UNSAT, UNKWN = process("pickle_16_01_2021/ALL")

# There is no point in trying to classify solving time/Hardness
# wrt SAT/UNSAT. The classifier will figure it out.  We can still do
# hardness classification for the SAT instances by looking at the meadian.
hardnessSAT = {
    # The distribution is bi-modal with a valley 
    # starting at around 2000 seconds. Therefore, 
    # it doesn't make sense to have more than two
    # bins.
    "easy"      : SAT.solvingTime.median(), 
    "hard"      : np.log2(np.inf)
}

def addSAThardness(dataframe, upperbounds):
    dataframe.insert(0, "SATHardness", np.digitize(dataframe["solvingTime"].values, upperbounds))
    return dataframe

# Recompute hardness labels for SAT instances
SAThardness = addSAThardness(SAT, list(hardnessSAT.values()))

# Run a classifier for category
# Check if the dataset is balanced or not. They grossly
# aren't. The best we can do at the moment is to use the whole dataset for category
# and run hardness classifier for SAT instances 
if (0):
	print("Are the datasets balanced for category?")
	print(np.unique(ALL.Category, 	return_counts=True))
	print(np.unique(SAT.Category, 	return_counts=True))
	print(np.unique(UNSAT.Category, return_counts=True))
	
	print("\nAre the datasets balanced for Hardness?")
	print(np.unique(ALL.Hardness, 	return_counts=True))
	print(np.unique(SAT.Hardness, 	return_counts=True))
	print(np.unique(SAT.SATHardness,	return_counts=True))
	print(np.unique(UNSAT.Hardness, return_counts=True))
	
	print("\nAre the datasets balanced for SAT/UNSAT?")
	print(np.unique(ALL.SAT, 	return_counts=True))
	print(np.unique(SAT.SAT, 	return_counts=True))
	print(np.unique(UNSAT.SAT, return_counts=True))


F = extractfeatures(ALL)
L = extractlabels(ALL)
FSAT = extractfeatures(SAT)
LSAT = extractlabels(SAT)

if (0): 
    # Run a test classifier for category
    print("Category")
    RF(F, L["Category"])
    
    # Run a test classifier for hardness
    print("Hardness")
    RF(F, L["Hardness"])
    
    # Run a test classifier for SAT/UNSAT
    print("SAT/UNSAT")
    RF(F, L["SAT"])
    
    # Run a test classifier for hardness
    print("SAT Hardness")
    RF(FSAT, LSAT["SATHardness"])

# Find out the top 5 features that predict Category using drop feature importance
QC = Parallel(n_jobs=2, verbose=1, prefer="threads")(delayed(RF)(F.filter(F5), L["Category"]) for F5 in list(combinations(F, 5)))
QC.sort(key = lambda l: l[1])
print("Top 5: category")
print("\t", QC[0])

# Find out the top 5 features that predict Hardness using drop feature importance
QH = Parallel(n_jobs=2, verbose=1, prefer="threads")(delayed(RF)(F.filter(F5), L["Hardness"]) for F5 in list(combinations(F, 5)))
QH.sort(key = lambda l: l[1])
print("Top 5: Hardness")
print("\t", QH[0])

# Find out the top 5 features that predict Hardness using drop feature importance
QSH = Parallel(n_jobs=2, verbose=1, prefer="threads")(delayed(RF)(FSAT.filter(F5), L["SATHardness"]) for F5 in list(combinations(F, 5)))
QSH.sort(key = lambda l: l[1])
print("Top 5: Hardness")
print("\t", QSH[0])
