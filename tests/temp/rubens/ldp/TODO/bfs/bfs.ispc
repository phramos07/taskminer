# include "ldp/graph.ispc"


// Traverses the matrix in a depth-first fashion
void dfs(uniform struct Graph& graph, uniform int root) {

	if (graph.node[root].visited) return;
	graph.node[root].visited = true;

	foreach (i = 0 ... graph.node[root].length) {

		int child = graph.node[root].edge[i].node;
		if (!graph.node[child].visited) ldp dfs(graph, child);

	}

}

export void graph_dfs(uniform struct Graph& graph, uniform int root) {
	foreach (i = 0 ... graph.num_nodes) graph.node[i].visited = false;
	dfs(graph, root);
}

// Traverses the matrix in a depth-first fashion (no-LDP)
void dfs_par(uniform struct Graph& graph, uniform int root) {

	if (graph.node[root].visited) return;
	graph.node[root].visited = true;

	foreach (i = 0 ... graph.node[root].length) {

		int child = graph.node[root].edge[i].node;
		if (!graph.node[child].visited) {

			for (uniform int j = 0; j < programCount; ++j) {

				// If thread is active within the loop,
				// node child_i must be visited
				if (programIndex == j) {
					uniform int child_i = extract(child, j);
					dfs(graph, child_i);
				}

			}

		}

	}

}

export void graph_dfs_par(uniform struct Graph& graph, uniform int root) {

	foreach (i = 0 ... graph.num_nodes) graph.node[i].visited = false;
	dfs_par(graph, root);

}
