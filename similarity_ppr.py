import networkx as nx
#Defining a directed graph
G = nx.DiGraph()
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
