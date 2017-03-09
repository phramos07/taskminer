# include <cstring>
# include <iomanip>
# include <iostream>

# include "../../timing.h"
# include "ldp/graph.hpp"
# include "ldp/common.hpp"


// Traverses the matrix in a depth-first fashion (no-LDP)
void dfs(struct Graph& graph, int root, float *f) {

	if (graph.node[root].visited) return;
	graph.node[root].visited = true;

	// Eventual computations
	f[root] = graph.node[root].length / float(graph.num_nodes);

	// Traversal
	for (int i = 0; i < graph.node[root].length; ++i) {

		int child = graph.node[root].edge[i].node;
		if (!graph.node[child].visited) dfs(graph, child, f);

	}

}

void graph_dfs(struct Graph& graph, int root, float *f) {

	for (int i = 0; i < graph.num_nodes; ++i) graph.node[i].visited = false;
	dfs(graph, root, f);

}

int main (int argc, char *argv[]) {

	int num_nodes, num_edges;
	std::cin >> num_nodes >> num_edges;

	int *distance = new int[num_nodes * num_nodes];
	struct Node *node = new struct Node[num_nodes];
	struct Edge *edge = new struct Edge[num_edges];

	int count(0);
	int u, v, weight, last = -1;
	while (count < num_edges and std::cin >> u >> v >> weight) {

		while (last < u) {
			++last;
			node[last].length = 0;
			node[last].edge = NULL;
			node[last].visited = false;
		}

		if (node[u].length == 0) node[u].edge = &edge[count];

		edge[count].node = v;
		edge[count].weight = weight;

		++node[u].length;
		++count;

	}

	double elapsed(0);

	// Traversing
	struct Graph graph;
	graph_build(graph, node, num_nodes, edge, num_edges, distance);
	// graph_dump(graph);

	float *f = new float[num_nodes];

	reset_and_start_timer();
	for (int i(0); i < num_nodes; ++i) graph_dfs(graph, i, f);
	elapsed += get_elapsed_mcycles();

	// graph_dump(graph);
	print_runtime(elapsed);

	delete[] f;
	delete[] node;
	delete[] edge;
	delete[] distance;

	return 0;

}
