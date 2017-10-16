#!/usr/bin/python


from sys import argv
from random import randint

minNode = 0
maxNode = 50

curX = 0
curY = 0

cur_id = 0
cur_name = "0_0"
table = {cur_name : cur_id}

edges = []

num_nodes = int(argv[1])
while len(table) < num_nodes:

	newX = randint(minNode, maxNode)
	newY = randint(minNode, maxNode)
	new_name = str(curX) + "_" + str(curY)

	if not (new_name in table):
		table[new_name] = len(table) + 1
	new_id = table[new_name]

	edges.append("\t".join([str(cur_id), str(new_id), str(randint(0, 100))]))

	curX = newX
	curY = newY

	cur_id = new_id
	cur_name = new_name

print(len(table))
print(len(edges))
for i in edges: print i
