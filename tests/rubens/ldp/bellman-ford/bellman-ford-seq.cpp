# include <cstring>
# include <iomanip>
# include <iostream>

# include "ldp/graph.hpp"
# include "ldp/common.hpp"
# include "../../timing.h"


// Bellman-Ford shortest paths
void relax_edges(struct Graph& graph, int u_id, int k_id) {

	Node& u = graph.node[u_id];
	Node& k = graph.node[k_id];

	int dist_uk = u.distance[k_id];

	// Relaxing edges u -> k -> v
	for (int v = 0; v < k.length; ++v) {

		int v_id = k.edge[v].node;
		int dist_kv = k.edge[v].weight;

		if (dist_uk != INF and dist_uk + dist_kv < u.distance[v_id]) {
			u.distance[v_id] = dist_uk + dist_kv;
		}

	}

}

void bellman_ford(struct Graph& graph, int root) {

	graph.node[root].distance[root] = 0;
	for (int i = 0; i < graph.num_nodes - 1; ++i) {
		for (int j = 0; j < graph.num_nodes; ++j) {
			if (graph.node[j].length > 0) relax_edges(graph, root, j);
		}
	}

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

	// Traversing
	struct Graph graph;
	graph_build(graph, node, num_nodes, edge, num_edges, distance);
	// graph_dump(graph);

	reset_and_start_timer();
	for (int i(0); i < num_nodes; ++i) bellman_ford(graph, i);
	double elapsed = get_elapsed_mcycles();

	// graph_dump(graph);
	print_runtime(elapsed);

	delete[] node;
	delete[] edge;
	delete[] distance;

	return 0;

}
