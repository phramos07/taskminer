# define INF (1 << 29)

struct Edge {

	uniform int node;
	uniform int weight;

};

struct Node {

	uniform bool visited;

	uniform int length;
	struct Edge * uniform edge;

	int * uniform distance;

};

struct Graph {

	uniform int num_nodes;
	uniform int num_edges;

	struct Node * uniform node;
	struct Edge * uniform edge;

};


// Builds the graph
export void graph_build(uniform struct Graph& graph,
	struct Node * uniform node, const uniform int& num_nodes,
	struct Edge * uniform edge, const uniform int& num_edges,
	int * uniform distance) {

	graph.node = node;
	graph.num_nodes = num_nodes;

	graph.edge = edge;
	graph.num_edges = num_edges;

	for (uniform int i = 0; i < num_nodes; ++i) {
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

	for (uniform int i = 0; i < graph.num_nodes; ++i) {
		print ("\n    ");
		for (uniform int j = 0; j < graph.num_nodes; ++j) {
			uniform int dist = graph.node[i].distance[j];
			print ("% ", (dist == INF ? -1 : dist));
		}
	}

	print ("\n\n}\n");

}

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
