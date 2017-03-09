# include <cstring>
# include <iomanip>
# include <iostream>

# include "ldp/graph.hpp"
# include "ldp/common.hpp"
# include "../../timing.h"


// Retrieves each connected component its maximum rank in a depth-first fashion
void cc_leader(int *rank, struct Graph& graph, int root) {

	if (graph.node[root].visited) return;
	graph.node[root].visited = true;

	for (int i = 0; i < graph.node[root].length; ++i) {

		int child = graph.node[root].edge[i].node;
		if (!graph.node[child].visited) {

			if (rank[root] > rank[child]) rank[child] = rank[root];
			cc_leader(rank, graph, child);

		}

	}

}

void graph_cc_leader(int * rank, struct Graph& graph) {

	for (int i = 0; i < graph.num_nodes; ++i) rank[i] = i;

	for (int i = 0; i < graph.num_nodes; ++i) {

		for (int j = 0; j < graph.num_nodes; ++j) {
			graph.node[j].visited = false;
		}

		if (!graph.node[i].visited) cc_leader(rank, graph, i);

	}

}

int main (int argc, char *argv[]) {

	int num_nodes, num_edges;
	std::cin >> num_nodes >> num_edges;

	int *rank = new int[num_nodes];
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

	// Traversing
	struct Graph graph;
	graph_build(graph, node, num_nodes, edge, num_edges, NULL);
	// graph_dump(graph);

	reset_and_start_timer();
	graph_cc_leader(rank, graph);
	double elapsed = get_elapsed_mcycles();

	// for (int i = 0; i < num_nodes; ++i) std::cout << rank[i] << " ";
	// std::cout << std::endl;

	print_runtime(elapsed);

	delete[] node;
	delete[] edge;
	delete[] rank;

	return 0;

}
