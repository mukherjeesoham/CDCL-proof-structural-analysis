import numpy as np
import scipy
import sklearn
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, svm, metrics
from SATmain import hardness
from SATclassifier import collate, load, extract 
from sklearn.inspection import permutation_importance

# TODO: For today's meeting
# [1] We want to understand which of the features is contributing the most to 
# the classification results, both for category and hardness, but especially for category. 
# [2] Define hardness in terms of the quartiles so that we have almost the same number in each 
# of the sets. [Done. It doens't help improve the results of the classifier.]
#   Look at the confusion matrix.
# [3] Colour tSNE plots with different metrics and see if we see any patterns 
# that are worth exploring.  
# [4] Why is the accuracy for the categories, so high? What if we restrict the dataset to just the 
# 9 parameters; how well would our classifier do? 

filename = "pickle_31_12_2020" 
dataframe = collate(load("../../datasets/{}".format(filename)), hardness)
# Read in data for category [Extract has been modified to balance the data.]
clabels, cfeatures, columnheaders = extract(dataframe, "Category")
cfeatures = StandardScaler().fit_transform(cfeatures) 
# Read in data for hardness
hlabels, hfeatures, columnheaders = extract(dataframe, "Hardness")
hfeatures = StandardScaler().fit_transform(hfeatures) 

# Choose the classifier as RandomForest. Worry about the hyperparameters later.
clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1) 

# Look at the confusion matrix for category
cftrain, cftest, cltrain, cltest = train_test_split(cfeatures, clabels, random_state=0)
clf.fit(cftrain, cltrain)
print("\n--------------------------------")
print("Category")
print("--------------------------------")
print("RF train accuracy: %0.3f" % clf.score(cftrain, cltrain))
print("RF test accuracy: %0.3f" % clf.score(cftest, cltest))
plot_confusion_matrix(clf, cftest, cltest)  
plt.show()  

# Look at permutation importance (one of the measures of feature importance)
result = permutation_importance(clf, cftest, cltest, n_repeats=10, random_state=42, n_jobs=2)
sorted_idx  = result.importances_mean.argsort()
topfeatures = columnheaders[sorted_idx].to_numpy()
print(topfeatures[:5])

# Look at the confusion matrix for hardness
hftrain, hftest, hltrain, hltest = train_test_split(hfeatures, hlabels, random_state=0)
clf.fit(hftrain, hltrain)
print("--------------------------------")
print("Hardness")
print("--------------------------------")
print("RF train accuracy: %0.3f" % clf.score(hftrain, hltrain))
print("RF test accuracy: %0.3f" % clf.score(hftest, hltest))
plot_confusion_matrix(clf, hftest, hltest)  
plt.show()

result = permutation_importance(clf, hftest, hltest, n_repeats=10, random_state=42, n_jobs=2)
sorted_idx  = result.importances_mean.argsort()
topfeatures = columnheaders[sorted_idx].to_numpy()
print(topfeatures[:5])


