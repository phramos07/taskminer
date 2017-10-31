# define INF (1 << 29)

struct Edge {

	int node;
	int weight;

};

struct Node {

	bool visited;

	int length;
	struct Edge *edge;

	int *distance;

};

struct Graph {

	int num_nodes;
	int num_edges;

	struct Node *node;
	struct Edge *edge;

};


// Builds the graph
void graph_build(struct Graph& graph,
	struct Node *node, const int& num_nodes,
	struct Edge *edge, const int& num_edges,
	int *distance) {

	graph.node = node;
	graph.num_nodes = num_nodes;

	graph.edge = edge;
	graph.num_edges = num_edges;

	for (int i = 0; i < num_nodes; ++i) {

		if (distance == NULL) {
			graph.node[i].distance = NULL;
			continue;
		}

		graph.node[i].distance = &distance[i * num_nodes];
		for (int j = 0; j < num_nodes; ++j) {
			graph.node[i].distance[j] = INF;
		}

	}

}

// Dumps the graph
void graph_dump(struct Graph& graph) {

	std::cout << "\nGraph G(V, E), |V| = " << graph.num_nodes
		<< ", |E| = " << graph.num_edges << " {\n";

	for (int i = 0; i < graph.num_nodes; ++i) {
		std::cout << "\n    ";
		int k = 0;
		for (int j = 0; j < graph.num_nodes; ++j) {
			if (k < graph.node[i].length && j == graph.node[i].edge[k].node) {
				std::cout << j << " "; ++k;
			} else std::cout << "- ";
		}
	}

	std::cout << "\n";

	for (int i = 0; i < graph.num_nodes; ++i) {
		std::cout << "\n    ";
		int k = 0;
		for (int j = 0; j < graph.num_nodes; ++j) {
			if (k < graph.node[i].length && j == graph.node[i].edge[k].node) {
				std::cout << graph.node[i].edge[k].weight << " "; ++k;
			} else std::cout << "-- ";
		}
	}

	std::cout << "\n";
	if (graph.num_nodes && graph.node[0].distance == NULL) {
		std::cout << "\n}\n";
		return;
	}

	for (int i = 0; i < graph.num_nodes; ++i) {
		std::cout << "\n    ";
		for (int j = 0; j < graph.num_nodes; ++j) {
			int dist = graph.node[i].distance[j];
			std::cout << (dist == INF ? -1 : dist) << " ";
		}
	}

	std::cout << "\n\n}\n";

}
