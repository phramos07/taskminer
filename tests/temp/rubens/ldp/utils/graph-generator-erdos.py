#!/usr/bin/python


from sys import argv, exit
from random import randint
from networkx import erdos_renyi_graph

if len(argv) != 3:
	print("%s <num_vertices:int> <probability:int>" % (argv[0]))
	exit()

num_vertices = int(argv[1])
probability = float(argv[2]) / 100

graph = erdos_renyi_graph(num_vertices, probability, directed=True)
edges = graph.edges()

print(num_vertices)
print(len(edges))

for (u, v) in sorted(edges):
	print("%d\t%d\t%d" % (u, v, randint(0, 100)))
