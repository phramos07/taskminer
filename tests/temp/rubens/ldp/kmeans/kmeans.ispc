# include "crev/vector.ispc"

struct Point {

	int label;
	int cluster;
	int old_cluster;

	vector_int dist;
	vector_int coord;

};

crev_vector(Point);


uniform bool changed_cluster(const uniform struct vector_point& point) {

	bool has_changed = false;
	foreach (i = 0 ... point.length) {
		has_changed |= (point.data[i].cluster != point.data[i].old_cluster);
	}

	return any(has_changed);

}

uniform float
distance(const uniform struct Point& a, const uniform struct Point& b) {

	const uniform struct vector_int& ca = a.coord;
	const uniform struct vector_int& cb = b.coord;

	float dist = 0;
	foreach (i = 0 ... ca.length) {
		dist += (ca.data[i] - cb.data[i]) * (ca.data[i] - cb.data[i]);
	}

	return reduce_add(dist);

}

void assign_centroid(uniform struct vector_Point& point,
	uniform struct vector_Point& centroid) {

	for (uniform int i = 0; i < point.length; ++i) {

		uniform int min_id = -1;
		uniform float min_dist = 999999999;

		for (uniform int j = 0; j < centroid.length; ++j) {

			uniform struct Point& c = centroid.data[j];
			if (min_dist > c.dist.data[i]) {
				min_id = j;
				min_dist = c.dist.data[i];
			}

		}

		point.data[i].cluster = min_id;

	}

}

void update_centroid(uniform struct vector_Point& point,
	uniform struct vector_Point& centroid) {
	...;
}

export void kmeans_crev(uniform struct vector_Point& point,
	uniform struct vector_Point& centroid) {

	while (changed_cluster(point)) {

		// Computing distances
		for (uniform int i = 0; i < centroid.length; ++i) {

			uniform struct Point& c = centroid.data[i];
			for (uniform int j = 0; j < point.length; ++j) {
				c.dist.data[i] = distance(c, point.data[j]);
			}

		}

		// Determining nearest centroid to each point
		assign_centroid(point, centroid);

		// Updating centroids
		update_centroid(point, centroid);

	}

}
