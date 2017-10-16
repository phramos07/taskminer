# include <ctime>
# include <cstdlib>
# include <iostream>

# include "sort_ispc.h"
# include "../../timing.h"
# include "ldp/common.hpp"

# define INF (1 << 29)


void set_order(int *array, int len) {

	if (len == 2 && array[0] > array[1]) {
		int aux = array[1];
		array[1] = array[0];
		array[0] = aux;
	} else if (len > 2) {
		printf("ERROR!\n");
	}

}

void mergesort_seq(int *array, int len) {

	if (len <= 2) {
		set_order(array, len);
		return;
	}

	int q = len / 2, r = len % 2;
	mergesort_seq(array, q);
	mergesort_seq(array + q, q + r);

	int *sorted = new int[len];
	int i = 0, j = q;

	// Merging subarrays
	for (int k = 0; k < len; ++k) {

		if (i < q && j < len) {

			if (array[i] < array[j]) {
				sorted[k] = array[i]; ++i;
			} else {
				sorted[k] = array[j]; ++j;
			}

		} else if (i < q) {
			sorted[k] = array[i]; ++i;
		} else {
			sorted[k] = array[j]; ++j;
		}

	}

	for (int p = 0; p < len; ++p) array[p] = sorted[p];
	delete[] sorted;

}

void quicksort_seq(int *array, int len) {

	if (len <= 2) {
		set_order(array, len);
		return;
	}

	int aux;
	int mid = len - 2;
	int pivot = array[len - 1];

	int i = 0;
	while (i < mid) {

		if (array[i] <= pivot) ++i;
		else {
			aux = array[mid];
			array[mid] = array[i];
			array[i] = aux;
			--mid;
		}

	}

	if (array[mid] <= pivot) mid = mid + 1;
	aux = array[mid];
	array[mid] = array[len - 1];
	array[len - 1] = aux;

	quicksort_seq(array, mid);
	quicksort_seq(array + mid + 1, (len - mid) - 1);

}


int main (int argc, char *argv[]) {

	int num_elements;
	std::cin >> num_elements;

	num_elements = num_elements * 1024;
	int *array = new int[num_elements];
	int *tmp = new int[num_elements];

// 	std::srand(std::time(0));
	for (int i = 0; i < num_elements; ++i) {
		array[i] = rand() % 100000;
		tmp[i] = array[i];
	}

	// Running mergesort
	double elapsed(0);

	reset_and_start_timer();
	ispc::mergesort_ldp(array, num_elements);

	std::cerr << "merge-ldp-__ ";
	elapsed = get_elapsed_mcycles();
	print_runtime(elapsed);

// for (int i = 0; i < num_elements; ++i) std::cout << array[i] << ", ";
// std::cout << std::endl;

	// Running quicksort
	for (int i = 0; i < num_elements; ++i) array[i] = tmp[i];

	reset_and_start_timer();
	ispc::quicksort_ldp(array, num_elements);

	std::cerr << "quick-ldp-__ ";
	elapsed = get_elapsed_mcycles();
	print_runtime(elapsed);

// for (int i = 0; i < num_elements; ++i) std::cout << array[i] << ", ";
// std::cout << std::endl;

	// ------------------------------------------------------------------------
	// Using bitonic sort for length < 8 

	// Running mergesort
	for (int i = 0; i < num_elements; ++i) array[i] = tmp[i];

	reset_and_start_timer();
	ispc::mergesort_ldp_bi(array, num_elements);

	std::cerr << "merge-ldp-bi ";
	elapsed = get_elapsed_mcycles();
	print_runtime(elapsed);

// for (int i = 0; i < num_elements; ++i) std::cout << array[i] << ", ";
// std::cout << std::endl;

	// Running quicksort
	for (int i = 0; i < num_elements; ++i) array[i] = tmp[i];

	reset_and_start_timer();
	ispc::quicksort_ldp_bi(array, num_elements);

	std::cerr << "quick-ldp-bi ";
	elapsed = get_elapsed_mcycles();
	print_runtime(elapsed);

// for (int i = 0; i < num_elements; ++i) std::cout << array[i] << ", ";
// std::cout << std::endl;

	// ------------------------------------------------------------------------
	// Using launch

	// Running mergesort
	for (int i = 0; i < num_elements; ++i) array[i] = tmp[i];

	reset_and_start_timer();
	ispc::mergesort_lch(array, num_elements);

	std::cerr << "merge-lch-__ ";
	elapsed = get_elapsed_mcycles();
	print_runtime(elapsed);

// for (int i = 0; i < num_elements; ++i) std::cout << array[i] << ", ";
// std::cout << std::endl;

	// Running quicksort
	for (int i = 0; i < num_elements; ++i) array[i] = tmp[i];

	reset_and_start_timer();
	ispc::quicksort_lch(array, num_elements);

	std::cerr << "quick-lch-__ ";
	elapsed = get_elapsed_mcycles();
	print_runtime(elapsed);

// for (int i = 0; i < num_elements; ++i) std::cout << array[i] << ", ";
// std::cout << std::endl;

	// ------------------------------------------------------------------------
	// Using launch and bitonic sort

	// Running mergesort
	for (int i = 0; i < num_elements; ++i) array[i] = tmp[i];

	reset_and_start_timer();
	ispc::mergesort_lch_bi(array, num_elements);

	std::cerr << "merge-lch-bi ";
	elapsed = get_elapsed_mcycles();
	print_runtime(elapsed);

// for (int i = 0; i < num_elements; ++i) std::cout << array[i] << ", ";
// std::cout << std::endl;

	// Running quicksort
	for (int i = 0; i < num_elements; ++i) array[i] = tmp[i];

	reset_and_start_timer();
	ispc::quicksort_lch_bi(array, num_elements);

	std::cerr << "quick-lch-bi ";
	elapsed = get_elapsed_mcycles();
	print_runtime(elapsed);

// for (int i = 0; i < num_elements; ++i) std::cout << array[i] << ", ";
// std::cout << std::endl;

	// ------------------------------------------------------------------------
	// Sequential

	// Running mergesort
	for (int i = 0; i < num_elements; ++i) array[i] = tmp[i];

	reset_and_start_timer();
	mergesort_seq(array, num_elements);

	std::cerr << "merge-seq-__ ";
	elapsed = get_elapsed_mcycles();
	print_runtime(elapsed);

// for (int i = 0; i < num_elements; ++i) std::cout << array[i] << ", ";
// std::cout << std::endl;

	// Running quicksort
	for (int i = 0; i < num_elements; ++i) array[i] = tmp[i];

	reset_and_start_timer();
	quicksort_seq(array, num_elements);

	std::cerr << "quick-seq-__ ";
	elapsed = get_elapsed_mcycles();
	print_runtime(elapsed);

// for (int i = 0; i < num_elements; ++i) std::cout << array[i] << ", ";
// std::cout << std::endl;

	delete[] array;
	delete[] tmp;
	return 0;

}
