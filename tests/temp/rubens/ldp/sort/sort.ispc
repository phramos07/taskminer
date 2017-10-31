static int bitonic_mask[6][2][8] = {

	{ {0, -1, -2, 3, 4, -5, -6, 7}, {1, 0, 3, 2, 5, 4, 7, 6} },
	{ {0, 1, -2, -3, -4, -5, 6, 7}, {2, 3, 0, 1, 6, 7, 4, 5} },
	{ {0, -1, 2, -3, -4, 5, -6, 7}, {1, 0, 3, 2, 5, 4, 7, 6} },

	{ {0, 1, 2, 3, -4, -5, -6, -7}, {4, 5, 6, 7, 0, 1, 2, 3} },
	{ {0, 1, -2, -3, 4, 5, -6, -7}, {2, 3, 0, 1, 6, 7, 4, 5} },
	{ {0, -1, 2, -3, 4, -5, 6, -7}, {1, 0, 3, 2, 5, 4, 7, 6} }

};

void set_order(int * uniform array, uniform int len) {

	if (len == 2 && array[0] > array[1]) {
		uniform int aux = array[1];
		array[1] = array[0];
		array[0] = aux;
	} else if (len > 2) {
		print("ERROR!\n");
	}

}

void bitonic_sort(int * uniform array, uniform int len) {

	if (len < 2) return;
	if (len == 2 && array[0] > array[1]) {
		uniform int aux = array[1];
		array[1] = array[0];
		array[0] = aux;
	}

	if (len > programCount) {
		print("ERROR!\n");
		return;
	}

	// Fetching values
	varying int val = (1 << 30), tmp = (1 << 30);
	if (programIndex < len) val = array[programIndex];

	// Bitonic sorting masks
	varying int cmp1 = bitonic_mask[0][0][programIndex];
	varying int cmp2 = bitonic_mask[1][0][programIndex];
	varying int cmp3 = bitonic_mask[2][0][programIndex];
                                      
	varying int val1 = bitonic_mask[0][1][programIndex];
	varying int val2 = bitonic_mask[1][1][programIndex];
	varying int val3 = bitonic_mask[2][1][programIndex];

	varying int cmp4 = bitonic_mask[3][0][programIndex];
	varying int cmp5 = bitonic_mask[4][0][programIndex];
	varying int cmp6 = bitonic_mask[5][0][programIndex];

	varying int val4 = bitonic_mask[3][1][programIndex];
	varying int val5 = bitonic_mask[4][1][programIndex];
	varying int val6 = bitonic_mask[5][1][programIndex];

	// Generating a bitonic sequence

	// Step 1: distance 1
	tmp = shuffle(val, val1);
	if (programIndex == cmp1) val = (val > tmp) ? tmp : val;
	else val = (val < tmp) ? tmp : val;

	// Step 2.1: distance 2
	tmp = shuffle(val, val2);
	if (programIndex == cmp2) val = (val > tmp) ? tmp : val;
	else val = (val < tmp) ? tmp : val;

	// Step 2.2: distance 4
	tmp = shuffle(val, val3);
	if (programIndex == cmp3) val = (val > tmp) ? tmp : val;
	else val = (val < tmp) ? tmp : val;

	// Sorting a bitonic sequence

	// Step 3.1: distance 1
	tmp = shuffle(val, val4);
	if (programIndex == cmp4) val = (val > tmp) ? tmp : val;
	else val = (val < tmp) ? tmp : val;

	// Step 3.2: distance 2
	tmp = shuffle(val, val5);
	if (programIndex == cmp5) val = (val > tmp) ? tmp : val;
	else val = (val < tmp) ? tmp : val;

	// Step 3.3: distance 4
	tmp = shuffle(val, val6);
	if (programIndex == cmp6) val = (val > tmp) ? tmp : val;
	else val = (val < tmp) ? tmp : val;

	// Storing sorted array
	if (programIndex < len) array[programIndex] = val;

}

// ----------------------------------------------------------------------------
// USING LDP

export void mergesort_ldp(int * uniform array, uniform int len) {

	if (len <= 2) {
		set_order(array, len);
		return;
	}

	uniform int q = len / 2, r = len % 2;

	int L[programCount] = {q, q + r};
	int * A[programCount] = {array, array + q};
	if (programIndex < 2) {
		ldp mergesort_ldp(A[programIndex], L[programIndex]);
	}

	uniform int * uniform sorted = uniform new int[len];
	uniform int i = 0, j = q;

	// Merging subarrays
	for (uniform int k = 0; k < len; ++k) {

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

	foreach (p = 0 ... len) array[p] = sorted[p];
	delete[] sorted;

}

export void quicksort_ldp(int * uniform array, uniform int len) {

	if (len <= 2) {
		set_order(array, len);
		return;
	}

	uniform int aux;
	uniform int mid = len - 2;
	uniform int pivot = array[len - 1];

	uniform int i = 0;
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

	int L[programCount] = {mid, (len - mid) - 1};
	int * A[programCount] = {array, array + mid + 1};
	if (programIndex < 2) {
		ldp quicksort_ldp(A[programIndex], L[programIndex]);
	}

}

// ----------------------------------------------------------------------------
// USING LDP + BITONIC SORT

export void mergesort_ldp_bi(int * uniform array, uniform int len) {

	if (len <= programCount) {
		bitonic_sort(array, len);
		return;
	}

	uniform int q = len / 2, r = len % 2;

	int L[programCount] = {q, q + r};
	int * A[programCount] = {array, array + q};
	if (programIndex < 2) {
		ldp mergesort_ldp_bi(A[programIndex], L[programIndex]);
	}

	uniform int * uniform sorted = uniform new int[len];
	uniform int i = 0, j = q;

	// Merging subarrays
	for (uniform int k = 0; k < len; ++k) {

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

	foreach (p = 0 ... len) array[p] = sorted[p];
	delete[] sorted;

}

export void quicksort_ldp_bi(int * uniform array, uniform int len) {

	if (len <= programCount) {
		bitonic_sort(array, len);
		return;
	}

	uniform int aux;
	uniform int mid = len - 2;
	uniform int pivot = array[len - 1];

	uniform int i = 0;
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

	int L[programCount] = {mid, (len - mid) - 1};
	int * A[programCount] = {array, array + mid + 1};
	if (programIndex < 2) {
		ldp quicksort_ldp_bi(A[programIndex], L[programIndex]);
	}

}

// ----------------------------------------------------------------------------
// USING LAUNCH

task void mergesort_launch(int * uniform array, uniform int len) {

	if (len <= 2) {
		set_order(array, len);
		return;
	}

	uniform int q = len / 2, r = len % 2;
	if (programIndex < q) launch mergesort_launch(array, q);
	if (programIndex < q + r) launch mergesort_launch(array + q, q + r);
	sync;

	// Merging subarrays
	uniform int * uniform sorted = uniform new int[len];

	uniform int a, b;
	uniform int j = 0, k = q, last = -1;
	for (uniform int i = 0; i < len; ++i) {

		if (j < q) a = array[j];
		if (k < len) b = array[k];

		if (j == q || k == len) {

			if (last == -1) {
				if (j == q) last = k;
				else last = j;
			}

			sorted[i] = array[last];
			++last;

		} else if (a <= b) {
			sorted[i] = a; ++j;
		} else {
			sorted[i] = b; ++k;
		}

	}

	foreach (i = 0 ... len) array[i] = sorted[i];
	delete[] sorted;

}

export void mergesort_lch(int * uniform array, uniform int len) {
	launch mergesort_launch(array, len);
}

task void quicksort_launch(int * uniform array, uniform int len) {

	if (len <= 2) {
		set_order(array, len);
		return;
	}

	uniform int aux;
	uniform int mid = len - 2;
	uniform int pivot = array[len - 1];

	uniform int i = 0;
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

	launch quicksort_launch(array, mid);
	launch quicksort_launch(array + mid + 1, (len - mid) - 1);
	sync;

}

export void quicksort_lch(int * uniform array, uniform int len) {
	launch quicksort_launch(array, len);
}

// ----------------------------------------------------------------------------
// USING LAUNCH AND BITONIC SORT

task void mergesort_launch_bi(int * uniform array, uniform int len) {

	if (len <= programCount) {
		bitonic_sort(array, len);
		return;
	}

	uniform int q = len / 2, r = len % 2;
	if (programIndex < q) launch mergesort_launch_bi(array, q);
	if (programIndex < q + r) launch mergesort_launch_bi(array + q, q + r);
	sync;

	// Merging subarrays
	uniform int * uniform sorted = uniform new int[len];

	uniform int a, b;
	uniform int j = 0, k = q, last = -1;
	for (uniform int i = 0; i < len; ++i) {

		if (j < q) a = array[j];
		if (k < len) b = array[k];

		if (j == q || k == len) {

			if (last == -1) {
				if (j == q) last = k;
				else last = j;
			}

			sorted[i] = array[last];
			++last;

		} else if (a <= b) {
			sorted[i] = a; ++j;
		} else {
			sorted[i] = b; ++k;
		}

	}

	foreach (i = 0 ... len) array[i] = sorted[i];
	delete[] sorted;

}

export void mergesort_lch_bi(int * uniform array, uniform int len) {
	launch mergesort_launch_bi(array, len);
}

task void quicksort_launch_bi(int * uniform array, uniform int len) {

	if (len <= programCount) {
		bitonic_sort(array, len);
		return;
	}

	uniform int aux;
	uniform int mid = len - 2;
	uniform int pivot = array[len - 1];

	uniform int i = 0;
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

	launch quicksort_launch_bi(array, mid);
	launch quicksort_launch_bi(array + mid + 1, (len - mid) - 1);
	sync;

}

export void quicksort_lch_bi(int * uniform array, uniform int len) {
	launch quicksort_launch_bi(array, len);
}
