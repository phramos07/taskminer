# include <cstring>
# include <iomanip>
# include <iostream>

# include "ldp/graph.hpp"
# include "../../timing.h"


// Traverses the matrix in a depth-first fashion (no-LDP)
void bfs(struct Graph& graph, int root) {

	if (graph.node[root].visited) return;
	graph.node[root].visited = true;

	for (int i = 0; i < graph.node[root].length; ++i) {

		int child = graph.node[root].edge[i].node;
		if (!graph.node[child].visited) dfs(graph, child);

	}

}

void graph_dfs(struct Graph& graph, int root) {

	for (int i = 0; i < graph.num_nodes; ++i) graph.node[i].visited = false;
	dfs(graph, root);

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
	graph_dump(graph);

	reset_and_start_timer();
	for (int i(0); i < num_nodes; ++i) graph_dfs(graph, i);
	elapsed += get_elapsed_mcycles();

	graph_dump(graph);
	print_runtime(elapsed);

	delete[] node;
	delete[] edge;
	delete[] distance;

	return 0;

}
