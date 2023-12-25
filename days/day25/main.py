import math

import networkx as nx
import matplotlib.pyplot as plt

with open("days/day25/input", "r") as f:
    line = f.read().splitlines()

G = nx.Graph()
for line in line:
    src, dests = line.split(": ")
    for dest in dests.split(" "):
        G.add_edge(src, dest)
        G.add_edge(dest, src)

fig = plt.figure()
nx.draw_networkx(G, ax=fig.add_subplot())
fig.savefig("out.png")

# Inspect out.png and hard code the edges specific to the input :D (its Christmas day...)
G.remove_edge("ffv", "mfs")
G.remove_edge("tbg", "ljh")
G.remove_edge("mnh", "qnv")

subgraphs = nx.connected_components(G)

print(f"part1: {math.prod(map(len, subgraphs))}")
