/*
void printer(int n) {
	print ("-- %\n", n);
}

void dfs(int(*function)(int), int n) {

	function(n);

}
*/

// All traversal data structures (matrix itself and visited-set)
// must be common to all threads:
// - parallelism is level-wise, which excludes the possibility of adding
//   edge (u, v) to the traversal multiple times
struct Matrix {

	int ** uniform data;
	uniform int num_nodes;

	uniform int level;
	bool * uniform visited;
	int ** uniform distance;

};

// Build matrix
export void matrix_build(uniform Matrix& matrix,
	int ** uniform data, uniform int num_nodes,
	int ** uniform distance) {

	matrix.data = data;
	matrix.num_nodes = num_nodes;

	matrix.level = 0;
	matrix.distance = distance;

	matrix.visited = uniform new bool[num_nodes];
	foreach (i = 0 ... matrix.num_nodes) matrix.visited[i] = false;

}

// Dump matrix
export void matrix_dump(uniform Matrix& matrix) {

	print ("\nMatrix(|V| = %) = {\n", matrix.num_nodes);

	for (uniform int i = 0; i < matrix.num_nodes; ++i) {

		print ("\n    ");
		for (uniform int j = 0; j < matrix.num_nodes; ++j) {
			print ("% ", matrix.data[i][j]);
		}

	}

	print ("\n\n}\n");

}

// Traverses the matrix in a depth-first fashion
void dfs(uniform Matrix& matrix, uniform int root) {

	if (matrix.visited[root]) return;
	matrix.visited[root] = true;

	print("\nINSISDE %\n", root);
	foreach (i = 0 ... matrix.num_nodes) {

		print("%: [%] ", matrix.level, matrix.data[root][i]);
		if (matrix.data[root][i] != INF) {
			if (!matrix.visited[i]) ldp dfs(matrix, i);
		}

	}

}

export void matrix_dfs(uniform Matrix& matrix, uniform int root) {
	foreach (i = 0 ... matrix.num_nodes) matrix.visited[i] = false;
	dfs(matrix, root);
}

// Bellman-Ford shortest paths algorithm implementation
void MATRIX_bellman_ford(uniform Matrix& matrix, uniform int distance[],
	uniform int root) {

	print("\nINSISDE %\n", root);
	foreach (i = 0 ... matrix.num_nodes) {

		print("%: [%] ", matrix.level, matrix.data[root][i]);
		if (distance[root] + matrix.data[root][i] < distance[i]) {
			distance[i] = distance[root] + matrix.data[root][i];
		}

	}

}

export void matrix_bellman_ford(uniform Matrix& matrix, uniform int root) {

	foreach (i = 0 ... matrix.num_nodes) matrix.distance[root][i] = INF;
	matrix.distance[root][root] = 0;

	for (uniform int i = 0; i < matrix.num_nodes - 1; ++i) {
		for (uniform j = 0; j < matrix.num_nodes; ++j) {
			if (matrix.distance[root][j] != INF) {
				ldp MATRIX_bellman_ford(matrix, matrix.distance[root], j);
			}
		}
	}

}

// Floyd-Warshall all-to-all shortest paths
export void matrix_floyd_warshall(uniform Matrix& matrix) {

	for (uniform int i = 0; i < matrix.num_nodes; ++i) {

		foreach (j = 0 ... matrix.num_nodes) matrix.visited[j] = false;
		foreach (j = 0 ... matrix.num_nodes) matrix.distance[i][j] = -1;

		// TODO implement it!

	}

}
