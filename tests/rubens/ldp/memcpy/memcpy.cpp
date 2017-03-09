# include <vector>
# include <iomanip>
# include <fstream>
# include <iostream>

# include "memcpy_ispc.h"
# include "../../timing.h"


int main (int argc, char *argv[]) {

	int n, num_threads;
	std::cin >> n >> num_threads;

	int *length = new int[n];
	int **src = new int*[n];
	int **dest = new int*[n];

	for (int i(0); i < n; ++i) {

		std::cin >> length[i];

		src[i] = new int[length[i]];
		dest[i] = new int[length[i]];

		std::srand(std::time(nullptr));
		for (int j(0); j < length[i]; ++j) src[j] = std::rand();

	}

	double elapsed(0);

	// Memory copy
	reset_and_start_timer();
	memcopy(src, dest, length, num_threads);

	std::cout << std::fixed, std::cout.precision(3);
	std::cout << "\n\n-- LDP[\"string\"]:\n    "
		<< elapsed << " million cycles" << std::endl;

	return 0;

}
