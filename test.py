
import numpy as np
import networkx as nx
import random
import pandas as pd
import matplotlib.pyplot as plt


def hill_climbing(net: nx.Graph, k: int, f):
    s = []
    nodes = net.nodes
    for t in range(k):
        temp = {}
        for node in nodes not in s:
            if node not in s:
                temp[node] = f([*s, node]) - f(s)
        s.append(max(temp, key=temp.get))

    return s


G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 2), (2, 3), (3, 4)])

if __name__ == '__main__':
    print(hill_climbing(G, 2, lambda x: 0))