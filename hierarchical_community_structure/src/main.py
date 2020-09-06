import igraph, sys, os, glob, time, joblib, GraphSAT 
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
        "savefig.dpi": 300,
        "text.usetex": True
})

def plotmodularity(path):
    # Compute graph from cnf
    g = GraphSAT.create_graph(GraphSAT.cnf_to_edges(GraphSAT.parse(path))) 

    # Compute community structure
    dendogram  = GraphSAT.compute_hierarchical_community_structure(GraphSAT.init(g))
    modularity = GraphSAT.compute_modularity(dendogram)

    # Plot modularity envelope
    depth = list(map(lambda x : x[0], modularity.vs["name"]))
    modularity = modularity.vs["modularity"] 
    plt.scatter(depth, modularity, s=1.0)
    plt.xlabel("$d$")
    plt.ylabel("$Q$")
    plt.title("%s" %path.split("/")[-2])
    filename = path.split("/")[-1].strip(".cnf")
    plt.savefig(f"../output/{filename}_modularity.pdf")
    plt.close()

paths =  glob.glob("../datasets/**/*.cnf", recursive=True)
joblib.Parallel(4, verbose=10)(joblib.delayed(plotmodularity)(path) for path in paths)

