// SIMD memory copy
void simd_memcopy(uniform int dest[], uniform int src[], uniform int n) {

	for (int i = programIndex; i < n; i += programCount) {
		dest[i] = src[i];
	}

}

// LDP-oriented SPMD memory copy
void opt_spmd_memcopy(int *dest, int *src, int n) {

	// For each worker "i" in the gang
	for (uniform int i = 0; i < programCount; ++i) {

		// If it was active at the callsite of this function
		if (programIndex == i) {

			// Broadcast parameters to all workers
			int64 uniform dest_tmp_i = extract((int64) dest, i);
			int64 uniform src_tmp_i = extract((int64) src, i);

			int * uniform dest_i = (int * uniform) dest_tmp_i;
			int * uniform src_i = (int * uniform) src_tmp_i;

			uniform int n_i = extract(n, i);

			unmasked {
				
				// Invoke memcopy with all threads
				simd_memcopy(dest_i, src_i, n_i);

			}

		}

	}

}

// SPMD memory copy with divergent threads kept sleeping
void spmd_memcopy(int *dest, int *src, int n) {

	// For each worker "i" in the gang
	for (uniform int i = 0; i < programCount; ++i) {

		// If it was active at the callsite of this function
		if (programIndex == i) {

			// Broadcast parameters to all workers
			int64 uniform dest_tmp_i = extract((int64) dest, i);
			int64 uniform src_tmp_i = extract((int64) src, i);

			int * uniform dest_i = (int * uniform) dest_tmp_i;
			int * uniform src_i = (int * uniform) src_tmp_i;

			uniform int n_i = extract(n, i);

			unmasked {
				
				// Invoke memcopy with all threads
				simd_memcopy(dest_i, src_i, n_i);

			}

		}

	}

}

// Copies data from input vectors to destination vectors
export void memcopy(int ** uniform dest, const int ** uniform src,
	int * uniform length, uniform int num_threads = 8) {

	if (programIndex < num_threads) {
		spmd_memcopy(dest, src, length);
	}

}
