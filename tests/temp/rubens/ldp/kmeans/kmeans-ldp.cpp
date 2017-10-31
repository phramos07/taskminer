# include <ctime>
# include <vector>
# include <cstdlib>
# include <iostream>

# include "kmeans_ispc.h"
# include "../../timing.h"
# include "ldp/common.hpp"


int main (int argc, char *argv[]) {

	int num_points, num_centroids, num_coords;
	std::cin >> num_points >> num_centroids >> num_coords;

	// Generating points
	ispc::vector_Point point;
	ispc::vector_Point_build(point, num_points);

	for (int i(0); i < num_points; ++i) {

		ispc::vector_Point_push_back(point, {i, -2, -1});

		ispc::Point& p(point.data[i]);
		ispc::vector_int_build(p.coord, num_coords);

		for (int j(0); j < num_coords; ++j) {
			ispc::vector_int_assign(p.coord, i, rand() % 100000);
		}

	}

	// Generating centroids
	ispc::vector_Point centroid;
	ispc::vector_Point_build(centroid, num_centroids);

	for (int i(0); i < num_centroids; ++i) {

		ispc::vector_Point_push_back(centroid, {i, i, i});

		ispc::Point& c(centroid.data[i]);
		ispc::vector_int_build(c.dist, num_points);
		ispc::vector_int_build(c.coord, num_coords);

		for (int j(0); j < num_coords; ++j) {
			ispc::vector_int_assign(c.coord, i, rand() % 100000);
		}

	}

	// Running kmeans CREV
	double elapsed(0);

	reset_and_start_timer();
	ispc::kmeans_crev(point, centroid);

	elapsed = get_elapsed_mcycles();
	print_runtime(elapsed);

	return 0;

}
