import networkx as nx
import pickle
import matplotlib.pyplot as plt

path = "../data/graphs.pkl"
with open(path, "rb") as handle:
    graphs = pickle.load(handle)

number = 26
n = 40

for j in range(1,7):
    k = round(1 + j * 0.1, 1)

    graph, article = graphs[(number, k)]
    sorted_weights_asc = sorted([graph.get_edge_data(*e)["weight"]
                                 for e in graph.edges], reverse=True)
    threshold = sorted_weights_asc[n]
    long_edges = list(filter(lambda e: e[2] < threshold,
                             (e for e in graph.edges.data('weight'))))
    le_ids = list(e[:2] for e in long_edges)
    graph.remove_edges_from(le_ids)
    for u,v in graph.edges:
        graph[u][v]['weight'] += 5

    to_remove = [node for node in graph.nodes if len(graph.edges(node))<= 1]
    graph.remove_nodes_from(to_remove)

    edges,weights = zip(*nx.get_edge_attributes(graph,'weight').items())
    pos = nx.circular_layout(graph)
    plt.figure()
    plt.title(f"k={k}, article no. {number}")
    nx.draw(graph, pos,
            node_color='b',
            edgelist=edges,
            edge_color=weights,
            with_labels=True,
            width=3.5,
            edge_cmap=plt.cm.Reds)
    plt.show()
print(article)
