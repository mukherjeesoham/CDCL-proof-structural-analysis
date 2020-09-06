from itertools import combinations
from collections import Counter
import igraph

def parse(path):
    f = open(path).read()
    g = f.split("\n")[1:-1]  
    h = list(map(lambda x: x.strip("0 "), g)) 
    clauses = list(map(lambda x: list(map(lambda y: abs(int(y)), x.split(" "))), h)) 
    m, n = list(map(lambda x : int(x), g[0].split(" ")[-2:]))
    return clauses

def clause_to_edges(clause):
    return set(combinations(list(map(lambda x:abs(int(x)) - 1, sorted(clause))), 2))

def cnf_to_edges(clauses):
    edge_set = set()
    for clause in clauses:
        edge_set = edge_set | clause_to_edges(clause) 
    return list(edge_set) 

def cnf_to_weighted_edges(clauses):
    edge_list = list()
    for clause in clauses:
        edge_list = edge_list + list(clause_to_edges(clause)) 
    assert all(map(lambda x : orderedPair(x), edge_list))
    return list(map(lambda x : x[0] + (x[1],), Counter(edge_list).items()))

def create_graph(edges):        
    if len(edges[0]) == 3:
        return igraph.Graph.TupleList(edges, edge_attrs="weight")
    else:
        return igraph.Graph.TupleList(edges)

def orderedPair(List):
    a, b = List
    if a <= b:
        return True
    else:
        return False

def init(graph):
    tree   = igraph.Graph()
    source = tree.add_vertex((0,0))
    source["graph"] = graph
    return source 
    
def compute_hierarchical_community_structure(source):
    membership = source["graph"].community_multilevel()
    if len(membership) > 1:
        for index, subgraph in enumerate(membership.subgraphs()):
            target = source.graph.add_vertex((source["name"][0] + 1, index))
            target["graph"] = subgraph 
            source.graph.add_edge(source, target)
            compute_hierarchical_community_structure(target) 
    return source.graph.as_directed()

def compute_modularity(dendogram):
    for vertex in dendogram.vs:
        vertex["modularity"] =  igraph.Graph.modularity(vertex["graph"], 
                                vertex["graph"].community_multilevel()) 
    return dendogram
        
