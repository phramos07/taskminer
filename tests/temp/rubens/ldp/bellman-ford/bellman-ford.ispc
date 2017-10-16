# include "ldp/graph.ispc"


// Bellman-Ford shortest paths
void relax_edges(uniform struct Graph& graph, uniform int u_id,
	uniform int k_id) {

	uniform Node& u = graph.node[u_id];
	uniform Node& k = graph.node[k_id];

	uniform int dist_uk = u.distance[k_id];

	// Relaxing edges u -> k -> v
	foreach (v = 0 ... k.length) {

		int v_id = k.edge[v].node;
		int dist_kv = k.edge[v].weight;

		if (dist_uk != INF && dist_uk + dist_kv < u.distance[v_id]) {
			u.distance[v_id] = dist_uk + dist_kv;
		}

	}

}

export void graph_bellman_ford(uniform struct Graph& graph, uniform int root) {

	graph.node[root].distance[root] = 0;
	for (uniform int i = 0; i < graph.num_nodes - 1; ++i) {
		foreach (j = 0 ... graph.num_nodes) {
			if (graph.node[j].length > 0) ldp relax_edges(graph, root, j);
		}
	}

}

export void graph_bellman_ford_par(uniform struct Graph& graph,
	uniform int root) {

	graph.node[root].distance[root] = 0;
	for (uniform int i = 0; i < graph.num_nodes - 1; ++i) {

		for (uniform int k_id = 0; k_id < graph.num_nodes; ++k_id) {

			if (graph.node[k_id].length <= 0) continue;

			uniform int u_id = root;
			uniform Node& u = graph.node[u_id];
			uniform Node& k = graph.node[k_id];

			uniform int dist_uk = u.distance[k_id];

			// Relaxing edges u -> k -> v
			foreach (v = 0 ... k.length) {

				int v_id = k.edge[v].node;
				int dist_kv = k.edge[v].weight;

				if (dist_uk != INF && dist_uk + dist_kv < u.distance[v_id]) {
					u.distance[v_id] = dist_uk + dist_kv;
				}

			}

		}

	}

}

/*
// Floyd-Warshall all-to-all shortest paths
export void graph_floyd_warshall(uniform struct Graph& graph) {

	for (uniform int src = 0; src < graph.num_nodes; ++src) {
		for (uniform int i = 0; i < graph.num_nodes; ++i) {
			foreach (j = 0 ... graph.num_nodes) {
				if (graph.node[j].length > 0) ldp relax_edges(graph, src, j);
			}
		}
	}

}
*/
