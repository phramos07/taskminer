# include "ldp/graph.ispc"


// Retrieves each connected component its maximum rank in a depth-first fashion
task void cc_leader_lch(int * uniform rank, uniform struct Graph& graph,
	uniform int root) {

	if (graph.node[root].visited) return;
	graph.node[root].visited = true;

	foreach (i = 0 ... graph.node[root].length) {

		int child = graph.node[root].edge[i].node;
		if (!graph.node[child].visited) {

			if (rank[root] > rank[child]) rank[child] = rank[root];

			for (uniform int j = 0; j < programCount; ++j) {

				// If thread is active within the loop,
				// node child_i must be visited
				if (programIndex == j) {
					uniform int child_i = extract(child, j);
					launch cc_leader_lch(rank, graph, child_i);
				}

			}

		}

	}

}

export void graph_cc_leader_lch(int * uniform rank,
	uniform struct Graph& graph) {

	foreach (i = 0 ... graph.num_nodes) rank[i] = i;

	for (uniform int i = 0; i < graph.num_nodes; ++i) {

		for (uniform int j = 0; j < graph.num_nodes; ++j) {
			graph.node[j].visited = false;
		}

		if (!graph.node[i].visited) launch cc_leader_lch(rank, graph, i);

	}

}

// Retrieves each connected component its maximum rank in a depth-first fashion
void cc_leader(int * uniform rank, uniform struct Graph& graph,
	uniform int root) {

	if (graph.node[root].visited) return;
	graph.node[root].visited = true;

	foreach (i = 0 ... graph.node[root].length) {

		int child = graph.node[root].edge[i].node;
		if (!graph.node[child].visited) {

			if (rank[root] > rank[child]) rank[child] = rank[root];
			ldp cc_leader(rank, graph, child);

		}

	}

}

export void graph_cc_leader(int * uniform rank, uniform struct Graph& graph) {

	foreach (i = 0 ... graph.num_nodes) rank[i] = i;

	for (uniform int i = 0; i < graph.num_nodes; ++i) {

		for (uniform int j = 0; j < graph.num_nodes; ++j) {
			graph.node[j].visited = false;
		}

		if (!graph.node[i].visited) cc_leader(rank, graph, i);

	}

}

// Retrieves each connected component its maximum rank in a depth-first fashion
void cc_leader_par(int * uniform rank, uniform struct Graph& graph,
	uniform int root) {

	if (graph.node[root].visited) return;
	graph.node[root].visited = true;

	foreach (i = 0 ... graph.node[root].length) {

		int child = graph.node[root].edge[i].node;
		if (!graph.node[child].visited) {

			if (rank[root] > rank[child]) rank[child] = rank[root];
	
			for (uniform int j = 0; j < programCount; ++j) {

				// If thread is active within the loop,
				// node child_i must be visited
				if (programIndex == j) {
					uniform int child_i = extract(child, j);
					cc_leader_par(rank, graph, child_i);
				}

			}

		}

	}

}

export void graph_cc_leader_par(int * uniform rank,
	uniform struct Graph& graph) {

	foreach (i = 0 ... graph.num_nodes) rank[i] = i;

	for (uniform int i = 0; i < graph.num_nodes; ++i) {

		for (uniform int j = 0; j < graph.num_nodes; ++j) {
			graph.node[j].visited = false;
		}

		if (!graph.node[i].visited) cc_leader_par(rank, graph, i);

	}

}
