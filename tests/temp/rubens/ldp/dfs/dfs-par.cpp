# include <cstring>
# include <iomanip>
# include <iostream>

# include "../../timing.h"
# include "dfs_ispc.h"
# include "ldp/common.hpp"

# define INF (1 << 29)


int main (int argc, char *argv[]) {

	int num_nodes, num_edges;
	std::cin >> num_nodes >> num_edges;

	int *distance = new int[num_nodes * num_nodes];
	struct ispc::Node *node = new struct ispc::Node[num_nodes];
	struct ispc::Edge *edge = new struct ispc::Edge[num_edges];

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
	struct ispc::Graph graph;
	ispc::graph_build(graph, node, num_nodes, edge, num_edges, distance);
	// ispc::graph_dump(graph);

	float *f = new float[num_nodes];

	reset_and_start_timer();
	for (int i(0); i < num_nodes; ++i) ispc::graph_dfs_par(graph, i, f);
	elapsed += get_elapsed_mcycles();

	// ispc::graph_dump(graph);
	print_runtime(elapsed);

	delete[] f;
	delete[] node;
	delete[] edge;
	delete[] distance;

	return 0;

}
