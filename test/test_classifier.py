#-----------------------------------------------------
# Test SATClassifier.py
# Soham 12/2020
#-----------------------------------------------------

import pytest
# FIXME: Add path
import SATclassifier as cls
import numpy as np


hardness = {
    "easy" : np.log2(60),
    "hard" : np.inf
}

def test_addhardness(): 
    data  = cls.collate(cls.load("../datasets/pickle"), hardness)
    hard  = data["Hardness"].values
    stime = data["solvingTime"].values 
    for (index, value) in enumerate(stime):
        if value > hardness["easy"]:
            assert(hard[index] ==  1)
        else:
            assert(hard[index] == 0)

def test_addcategory(): 
    data  = cls.collate(cls.load("../datasets/pickle"), hardness)
    cat   = data["Category"].values
    A, B  = np.unique(cat, return_counts=True)
    for index, dataframe in enumerate(cls.load("../datasets/pickle").values()):
        assert(B[index] == dataframe.shape[0])
        assert(A[index] == index)

def test_extract():
    data  = cls.collate(cls.load("../datasets/pickle"), hardness)
    hard  = data["Hardness"].values
    stime = data["solvingTime"].values
    labels, features = cls.extract(data, "Hardness") 
    assert(all(labels == hard))


