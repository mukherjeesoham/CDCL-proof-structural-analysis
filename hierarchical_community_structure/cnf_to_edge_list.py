def read_file(file):
	f=open("{0}".format(file),"r")
	#skip lines until header
	line = f.readline()
	line = line.split(" ")
	while line[0] != 'p' or line[1] != 'cnf':
		line = f.readline()
		line = line.split(" ")
	n = int(line[2])
	m = int(line[3])
	#store clauses in a list
	clauses=[]
	line = f.readline()
	while line:
	    l = line.split(" ")
	    clauses.append(l[0:-1])
	    line = f.readline()
	f.close()
	return clauses, m, n


def cnf_to_edge_list(clauses):
        edge_list = []
        for clause in clauses:
                for i in range(len(clause)-1):
                        for j in range(i+1, len(clause)):
                                edge_list.append([abs(int(clause[i]))-1, abs(int(clause[j]))-1])
        edge_set = set(map(frozenset, edge_list))
        return edge_set

# SM: Adding code to test cnf_to_edge_list, and then extending the function
# to compute a weighted adjacency matrix.   
from itertools import combinations
from collections import Counter

def clause_to_edge_set(clause):
    return set(combinations(clause, 2))

def cnf_to_edge_set(clauses)
    edge_set = set()
    for clause in clauses:
        edge_set = edge_set | clause_to_edge_set(clause) 
    return edge_set 

def cnf_to_edge_dict_weighted(clauses)
    # TODO: Think of other definitions for edge weights; for example, 
    # the simple association index.
    edge_list = list()
    for clause in clauses:
        edge_list = edge_list + list(clause_to_edge_set(clause)) 
    return dict(Counter(edge_set))

def cnf_to_weighted_graph(clauses)
    G = list()
    for (key, value) in cnf_to_edge_dict_weighted(clauses):
        source = key[1]
        target = key[2]
        weight = value
        G = G + [(source, target, weight)]
    return igraph.Graph.TupleList(G, edge_attrs="weight")
        
        









