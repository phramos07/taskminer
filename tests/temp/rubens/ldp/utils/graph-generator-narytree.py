#!/usr/bin/python


from sys import argv, exit
from random import randint

if len(argv) != 3:
	print("%s <arity:int> <height:int>" % (argv[0]))
	exit()

N = int(argv[1])
height = int(argv[2])

count = 0
edges = []

stack = [(0, 0)]
while stack:

	(label, index) = stack[-1]
	if index == N: stack.pop(); continue

	count += 1
	edges.append((label, count))

	stack[-1] = (label, index + 1)
	if len(stack) < height: stack.append((count, 0))

# Printing number of vertices, of edges, and edges
print(count + 1)
print(len(edges))
for (u, v) in sorted(edges):
	print("%d\t%d\t%d" % (u, v, randint(0, 100)))
