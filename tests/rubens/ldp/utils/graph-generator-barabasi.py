#!/usr/bin/python


from sys import argv, exit
from random import randint
from networkx import barabasi_albert_graph

if len(argv) != 3:
	print("%s <num_vertices:int> <outdegree:int>" % (argv[0]))
	exit()

num_vertices = int(argv[1])
outdegree = int(argv[2])

graph = barabasi_albert_graph(num_vertices, outdegree)
edges = graph.edges()

print(num_vertices)
print(len(edges))

for (u, v) in sorted(edges):
	print("%d\t%d\t%d" % (u, v, randint(0, 100)))
