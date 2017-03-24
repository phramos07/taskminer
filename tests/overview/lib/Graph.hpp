#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <iostream>
#include <list>
#include <vector>
#include <queue>


template<class NodeType, class EdgeType> struct Edge;
template<class NodeType, class EdgeType> struct Node;
template<class T> struct Coord;

template<class T>
struct Coord
{
	T x;
	T y;
};

template<class NodeType, class EdgeType>
struct Node 
{
	int index;
	NodeType weight;
	std::vector<Edge<NodeType, EdgeType>* > edges;
	bool visited;

	Node(int i, NodeType w) : index(i), weight(w) {};
	~Node() {};
	void addEdge(Edge<NodeType, EdgeType> &e) { edges.push_back(&e); };
};

template<class NodeType, class EdgeType>
struct Edge 
{
	Node<NodeType, EdgeType>* src;
	Node<NodeType, EdgeType>* dst;
	EdgeType weight;

	Edge(Node<NodeType, EdgeType>* s, Node<NodeType, EdgeType>* d, EdgeType w) :
		src(s), dst(d), weight(w) {};
	~Edge() {}
};

typedef int NT;
typedef int ET;

struct Graph 
{	
	std::vector<Node<NT, ET>* > nodes;
	std::list<Edge<NT, ET>* > edges;
	int size=0;

	Graph() {};
	Graph(int s);
	~Graph() {};
	void printToDot(std::ostream &o);
	Node<NT, ET>* operator[] (unsigned i);
	void readFromInput(std::istream &in);
	Graph* boruvka_MST();
	std::list<Node<NT, ET>* > shortestPath(Node<NT, ET> &src, Node<NT, ET> &dst);
	void relaxEdges(Node<NT, ET> &src, Node<NT, ET> &dst);
	void dfs();
	void dfs_visit(Node<NT, ET> &N);
	void bfs();
	void bfs_visit(Node<NT, ET> &N, std::queue<unsigned>& unexplored);
};

#endif