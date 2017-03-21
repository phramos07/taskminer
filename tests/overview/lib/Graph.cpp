#include "Graph.hpp"
#include <cmath>

static const int MAX_WEIGHT=1000;

Graph::Graph(int s) : size(s) 
{
	int weight;
	for (unsigned i = 0; i < size; i++)
	{
		Node<NT, ET>* node = new Node<NT, ET>(i, 0);
		nodes.push_back(node);
	}

	Node<NT, ET>* node1;
	Node<NT, ET>* node2;

	for (unsigned i = 0; i < size; i++)
		for (unsigned j = 0; j < size; j ++)
			if (j == i) continue;
			else
			{
				weight = rand()%MAX_WEIGHT;
				if (weight%10)
				{
					node1 = this->operator[](i);
					node2 = this->operator[](j);
					Edge<NT, ET>* edge = new Edge<NT, ET>(node1, node2, weight);
					edges.push_back(edge);
				}
			}
}

inline Node<NT, ET>* Graph::operator[] (unsigned i)
{
	// return nodes[i];
	for (auto &n : nodes)
	{
		if (n->index == i)
			return n;
	}

	return nullptr;
}

void Graph::printToDot(std::ostream &o)
{
  o << "digraph g {\n";
  for (auto &n : nodes)
  {
  	o << "\"" << n->index << "\";\n";
  }

  for (auto &e : edges)
  {
  	o << "\"" 
  		<< e->src->index 
  		<< "\" -> \"" 
  		<< e->dst->index 
  		<< "\" [label="
  		<< e->weight
  		<< "];\n";
  }

  o << "\n}";

}

void Graph::readFromInput(std::istream &in)
{
	int i1, i2, w;
	in >> i1 >> i2 >> w;
	Node<NT, ET>* n1 = new Node<NT, ET>(i1, 0);
	Node<NT, ET>* n2 = new Node<NT, ET>(i2, 0);
	size += 2;
	Edge<NT, ET>* e = new Edge<NT, ET>(n1, n2, w);

	nodes.push_back(n1);
	nodes.push_back(n2);
	edges.push_back(e);

	while (in >> i1)
	{
		in >> i2 >> w;
		n1 = this->operator[](i1);
		n2 = this->operator[](i2);
		if (!n1)
		{
			n1 = new Node<NT, ET>(i1, 0);
			nodes.push_back(n1);
			size++;
		}
		if (!n2)
		{
			n2 = new Node<NT, ET>(i2, 0);
			nodes.push_back(n2);
			size++;
		}

		e = new Edge<NT, ET>(n1, n2, w);
		edges.push_back(e);
	}
}

void Graph::dfs()
{
	for (auto &n : nodes)
		n->visited = false;

	for (auto &n : nodes)
		dfs_visit(*n);
}

void Graph::dfs_visit(Node<NT, ET> &N)
{
	N.visited = true;
	for (auto &e : N.edges)
		if (!e->dst->visited)
		{
			//Eventual computations
			//

			//recursive call
			dfs_visit(*e->dst);
		}
}

void Graph::bfs()
{
	for (auto &n : nodes)
		n->visited = false;

	std::queue<unsigned> unexplored;

	for (auto &n : nodes)
	{
		bfs_visit(*n, unexplored);
	}
}

void Graph::bfs_visit(Node<NT, ET> &N, std::queue<unsigned>& unexplored)
{
	if (!N.visited)
	{
		//eventual computations

		N.visited = true;

		for (auto &e : N.edges)
		{
			unexplored.push(e->dst->index);		
		}

		while (!unexplored.empty())
		{
			Node<NT, ET>* N1 = this->operator[](unexplored.front());
			unexplored.pop();
			bfs_visit(*N1, unexplored);
		}		
	}

}

