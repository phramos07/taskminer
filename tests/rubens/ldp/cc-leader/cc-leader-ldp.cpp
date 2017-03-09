# include <cstring>
# include <iomanip>
# include <iostream>

# include "../../timing.h"
# include "ldp/common.hpp"
# include "cc-leader_ispc.h"


int main (int argc, char *argv[]) {

	int num_nodes, num_edges;
	std::cin >> num_nodes >> num_edges;

	int *rank = new int[num_nodes];
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

	// Traversing
	struct ispc::Graph graph;
	ispc::graph_build(graph, node, num_nodes, edge, num_edges, NULL);
	// ispc::graph_dump(graph);

	reset_and_start_timer();
	ispc::graph_cc_leader(rank, graph);
	double elapsed = get_elapsed_mcycles();

	// for (int i = 0; i < num_nodes; ++i) std::cout << rank[i] << " ";
	// std::cout << std::endl;

	print_runtime(elapsed);

	delete[] node;
	delete[] edge;
	delete[] rank;

	return 0;

}
