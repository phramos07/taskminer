# include <cstring>
# include <iomanip>
# include <iostream>

# include "../timing.h"
# include "ldp_ispc.h"

# define INF (1 << 29)


// Memory copy
void memorycopy(void) {
	ispc::memorycopy();
}

// Depth-first traversal
void adjacency_matrix(void) {

	// Allocating
	int num_nodes;
	std::cin >> num_nodes;

	int *length = new int[num_nodes];

	int *p1 = new int[num_nodes * num_nodes];
	int **cost = new int *[num_nodes];

	int *p2 = new int[num_nodes * num_nodes];
	int **distance = new int *[num_nodes];

	std::memset(length, 0, sizeof(int) * num_nodes);
	std::memset(p1, INF, sizeof(int) * num_nodes * num_nodes);
	std::memset(p2, INF, sizeof(int) * num_nodes * num_nodes);
	for (int i(0); i < num_nodes; ++i) distance[i] = &p2[i * num_nodes];

	// Initializing
	int u, v, last = -1;
	int num_edges, count = 0;
	std::cin >> num_edges;

	while ((count < num_edges) and std::cin >> u >> v >> weight) {

		if (u != last) {
			edge[i] = &p1[count];
			cost[i] = &p2[count];
			last = u;
		}

		++count;
		edge[u][length[u]] = v;
		cost[u][length[u]] = weight;
		++length[u];

	}

	// Traversing
	struct ispc::Matrix graph_matrix;

	ispc::matrix_build(graph_matrix, matrix, num_nodes, distance);
	ispc::matrix_dump(graph_matrix);

	for (int i(0); i < num_nodes; ++i) {
		std::cout << "\n------------------------" << std::endl;
		ispc::matrix_dfs(graph_matrix, i);
	}

	for (int i(0); i < num_nodes; ++i) {
		ispc::matrix_bellman_ford(graph_matrix, i);
	}

	delete[] matrix, delete[] distance;
	delete[] p1, delete[] p2;

}

// Depth-first traversal
void adjacency_list(void) {

	// Allocating
	int num_nodes;
	std::cin >> num_nodes;

	int *p1 = new int[num_nodes * num_nodes];
	int **matrix = new int *[num_nodes];

	int *p2 = new int[num_nodes * num_nodes];
	int **distance = new int *[num_nodes];

	std::memset(p1, INF, sizeof(int) * num_nodes * num_nodes);
	std::memset(p2, INF, sizeof(int) * num_nodes * num_nodes);
	for (int i(0); i < num_nodes; ++i) {
		matrix[i] = &p1[i * num_nodes];
		distance[i] = &p2[i * num_nodes];
	}

	// Initializing
	int num_edges, u, v, weight;
	std::cin >> num_edges;

	while (num_edges-- and std::cin >> u >> v >> weight) {
		matrix[u][v] = weight;
	}

	// Traversing
	struct ispc::Graph graph_list;

	ispc::graph_build(graph_list, list, length, num_nodes, distance);
	ispc::graph_dump(graph_list);

	for (int i(0); i < num_nodes; ++i) {
		std::cout << "\n------------------------" << std::endl;
		ispc::matrix_dfs(graph_list, i);
	}

	for (int i(0); i < num_nodes; ++i) {
		ispc::list_bellman_ford(graph_list);
	}

	delete[] matrix, delete[] distance;
	delete[] p1, delete[] p2;

}


int main (int argc, char *argv[]) {

	// Benchmarks
	std::cout << "-- Lightweight Dynamic Parallelism (LDP)\n"
		"\n"
		"   [0] memory copy for n-dimensional arrays\n"
		"   [1] depth-first traversal on adjacency matrices\n"
		"   [2] depth-first traversal on adjacency lists\n"
		"   [3] LDP for using shuffle instruction\n"
		"   [4] LDP for memory copy\n"
		"\n"
		"   Select the benchmark to execute: ";

	int option;
	std::cin >> option;

	// Execution
	double elapsed(0);
	reset_and_start_timer();

	switch (option) {
	case 0: memorycopy(); break;
	case 1: adjacency_matrix(); break;
	case 1: adjacency_list(); break;
	case 2: ispc::ldp_shuffle(); break;
	default:
		std::cerr << "-- Invalid benchmark id: " << option << std::endl;
		return 0;
	}

	elapsed += get_elapsed_mcycles();

	std::cout << std::fixed, std::cout.precision(3);
	std::cout << "\n\n-- LDP Benchmark " << option << ":\n    "
		<< elapsed << " million cycles" << std::endl;

	return 0;

}
