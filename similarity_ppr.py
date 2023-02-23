import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import operator

#Defining a directed graph
G = nx.DiGraph()
# G = nx.bipartite.gnmk_random_graph(3, 5, 10, seed=123)
## Define a Bipartite Graph of people and items bought
G.add_nodes_from(["Alice", "Bob", "Charlie", "Diana",
                  "Orange", "Apple", "Banana", "Pineapple", "Raspberry"])
G.add_edges_from([
    ("Alice", "Orange"),
    ("Alice", "Apple"),
    ("Bob", "Orange"),
    ("Bob", "Apple"),
    ("Bob", "Banana"),
    ("Charlie", "Orange"),
    ("Charlie", "Apple"),
    ("Charlie", "Banana"),
    ("Diana", "Banana"),
    ("Diana", "Pineapple"),
    ("Diana", "Raspberry"),
])

top = nx.bipartite.sets(G)[0]
pos = nx.bipartite_layout(G, top)

ppr = nx.pagerank(G, personalization={"Alice": 1})
ppr = sorted(ppr.items(), key=operator.itemgetter(1), reverse=True)
for item, score in ppr:
    print(item, score)