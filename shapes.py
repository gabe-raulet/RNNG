import sys
import numpy as np
import tadasets
from dataset_io import *
import matplotlib.pyplot as plt

def distance(p, q):
    return np.linalg.norm(p - q)

def plot_points(points, g=None):
    n, d = points.shape
    assert d == 2
    plt.scatter(points[:,0], points[:,1], s=5, c="black")
    if g:
        for u in g:
            for v in g[u]:
                if u < v:
                    p = points[u]
                    q = points[v]
                    plt.plot([p[0], q[0]], [p[1], q[1]], 'black', linewidth=0.5)

def neighbor_graph(points, epsilon):
    n = len(points)
    graph = {u : [] for u in range(n)}
    for u in range(n):
        for v in range(u+1,n):
            d = distance(points[u], points[v])
            if d <= epsilon:
                graph[u].append(v)
                graph[v].append(u)
    return graph

circle = tadasets.dsphere(n=50, d=1, r=1, noise=0.1).astype(np.float32)
graph = neighbor_graph(circle, 1)
plot_points(circle, graph)
plt.show()

write_fvecs("circle.fvecs", circle)
