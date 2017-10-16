# define INF (1 << 29)


struct Vector {

	uniform int size;
	int * uniform data;

};


// Builds the graph
export void matrix_build(
export void graph_build(uniform struct Graph& graph,
	struct Node * uniform node, const uniform int& num_nodes,
	struct Edge * uniform edge, const uniform int& num_edges,
	int * uniform distance) {

	graph.node = node;
	graph.num_nodes = num_nodes;

	graph.edge = edge;
	graph.num_edges = num_edges;

	for (uniform int i = 0; i < num_nodes; ++i) {

		if (distance == NULL) {
			graph.node[i].distance = NULL;
			continue;
		}

		graph.node[i].distance = &distance[i * num_nodes];
		for (uniform int j = 0; j < num_nodes; ++j) {
			graph.node[i].distance[j] = INF;
		}

	}

}

// Dumps the graph
export void graph_dump(uniform struct Graph& graph) {

	print ("\nGraph G(V, E), |V| = %, |E| = % {\n",
		graph.num_nodes, graph.num_edges);

	for (uniform int i = 0; i < graph.num_nodes; ++i) {
		print ("\n    ");
		uniform int k = 0;
		for (uniform int j = 0; j < graph.num_nodes; ++j) {
			if (k < graph.node[i].length && j == graph.node[i].edge[k].node) {
				print ("% ", j); ++k;
			} else print ("- ");
		}
	}

	print ("\n");

	for (uniform int i = 0; i < graph.num_nodes; ++i) {
		print ("\n    ");
		uniform int k = 0;
		for (uniform int j = 0; j < graph.num_nodes; ++j) {
			if (k < graph.node[i].length && j == graph.node[i].edge[k].node) {
				print ("% ", graph.node[i].edge[k].weight); ++k;
			} else print ("-- ");
		}
	}

	print ("\n");
	if (graph.num_nodes && graph.node[0].distance == NULL) {
		print ("\n}\n");
		return;
	}

	for (uniform int i = 0; i < graph.num_nodes; ++i) {
		print ("\n    ");
		for (uniform int j = 0; j < graph.num_nodes; ++j) {
			uniform int dist = graph.node[i].distance[j];
			print ("% ", (dist == INF ? -1 : dist));
		}
	}

	print ("\n\n}\n");

}
