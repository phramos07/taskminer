#include "../lib/Graph.hpp"
#include <cmath>
#include <omp.h>

static const int SIZE = 10000;

Graph G(SIZE);

double dfs(Node<NT, ET>& N);

void graph_dfs();

int main(int argc, char const *argv[])
{
	graph_dfs();

	return 0;
}

void graph_dfs()
{
	for (unsigned i = 0; i < G.size; i++)
		G[i]->visited = false;

	dfs(*G[0]);
}

double dfs(Node<NT, ET>& N)
{
	double dist = 0.0;
	if (!N.visited)
	{
		N.visited = true;

		//Eventual computations
		dist += sqrt(pow(N.index + N.weight, 2) + pow(N.index + 4.0*N.weight, 2));

		//dfs recursive visit
		#pragma omp parallel
		#pragma omp single
		for (auto &e : N.edges)
		{
			int index = e->dst->index;
			#pragma omp task depend(inout:(*G)[index])
			dfs(*G[index]);
		}
	}

	return dist;
}

