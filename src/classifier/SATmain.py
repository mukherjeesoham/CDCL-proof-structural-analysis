#-----------------------------------------------------
# SAT classifier
# Soham 11/2020
# Here's the question we want to answer. Given several families SAT instances
# and their solving times on a specific CDCL solver, can we 
# - build a classifier that assigns a new SAT instance to one of the families? 
# - build a classifier that classifies a SAT instance (within a particular) family as easy or hard?
#-----------------------------------------------------

from SATclassifier import collate, classifier, load, hardness

def main(filename):
    data = collate(load("../../datasets/{}".format(filename)), hardness)
    print("----------------------------------")
    print("Classification results for Category")
    print("----------------------------------")
    classifier(data, "Category")
    print("----------------------------------")
    print("Classification results for Hardness")
    print("----------------------------------")
    classifier(data, "Hardness")

# main("pickle_31_12_2020")

