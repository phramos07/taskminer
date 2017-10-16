#ifndef __GRAPH_H__
#define __GRAPH_H__

static const int SIZE = 100;

struct Node
{
	int index;
	
}typedef Node;

struct Edge
{
	Node* src;
	Node* dst;
	int weight;
	
}typedef Edge;

struct Graph
{
	Node* nodes;
	Edge* edges;	

}typedef Graph;

Graph* newGraph(int size=SIZE);

void addEdge(int src, int dst, int weight);

#endif

