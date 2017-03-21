#include "../lib/Graph.hpp"

static const int SIZE = 100;

Graph G(SIZE);

void dfs(Node<NT, ET>& N);

void graph_dfs();

int main(int argc, char const *argv[])
{

	return 0;
}

void graph_dfs()
{
	for (unsigned i = 0; i < G.size; i++)
		G[i]->visited = false;

	dfs(*G[0]);
}

void dfs(Node<NT, ET>& N)
{
	if (!N.visited)
	{
		N.visited = true;

		//Eventual computations

		for (auto &e : N.edges)
		{
			int index = e->dst->index;
			dfs(*G[index]);
		}
	}
}

