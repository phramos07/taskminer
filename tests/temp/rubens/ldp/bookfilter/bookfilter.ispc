struct String {

	uniform int length;
	int8 * uniform data;

};


// ----------------------------------------------------------------------------
// Parallel implementation

// Parallel memory copy
void mem_cpy_par(uniform int8 * varying dest, const uniform int8 * varying src,
	const varying int n) {
	for (varying int i = 0; i < n; ++i) dest[i] = src[i];
}

// Matches the pattern with the input text
void str_match_par(const int8 * varying text, const int varying length,
	const uniform String pattern, varying bool& match) {

	for (varying int i = 0; i < length; ++i) {

		varying int j = 0;
		while ((i + pattern.length) <= length
			&& j < pattern.length
			&& text[i + j] == pattern.data[j]) {
			++j;
		}

		// Check if found
		if (j == pattern.length) {
			match = true;
			break;
		}

	}

}

// Copies only pages containing the input pattern
export void bookfilter_par(const String * uniform page,
	const uniform int num_pages, const uniform String& pattern,
	String * uniform output) {

	foreach (i = 0 ... num_pages) {

		bool match = false;
		str_match_par(page[i].data, page[i].length, pattern, match);

		if (match != false) {
			mem_cpy_par(output[i].data, page[i].data, page[i].length);
			output[i].length = page[i].length;
		}

	}

}

// ----------------------------------------------------------------------------
// LDP implementation

// SIMD memory copy
void mem_cpy_simd(uniform int8 * uniform dest, const uniform int8 * uniform src,
	const uniform int n) {
	foreach (i = 0 ... n) dest[i] = src[i];
}

// Matches the pattern with the input text
void str_match_simd(const int8 * uniform text, const uniform String pattern,
	const uniform int offset, uniform bool * uniform has_matched) {

	*has_matched = true;
	foreach (i = 1 ... pattern.length) {
		if (pattern.data[i] != text[i + offset]) *has_matched = false;
	}

}

// Copies only pages containing the input pattern
export void bookfilter_ldp(const struct String * uniform page,
	const uniform int num_pages, const uniform struct String& pattern,
	struct String * uniform output) {

	foreach (i = 0 ... num_pages) {

		// bool match[8] = {false, false, false, false, false, false, false, false};
//		for (varying int j = 0; j < page[i].length; ++j) {

//			if (page[i].data[j] == pattern.data[0]
//				&& (j + pattern.length) <= page[i].length) {

				// ldp str_match_simd(page[i].data, pattern, j, &match[programIndex]);
		bool match = false;
		str_match_par(page[i].data, page[i].length, pattern, match);

				if (match != false) {
					ldp mem_cpy_simd(output[i].data, page[i].data, page[i].length);
					output[i].length = page[i].length;
					//break;
				}

//			}

//		}

	}

}


// ----------------------------------------------------------------------------
// Launch implementation

// SIMD memory copy
task void task_mem_cpy_simd(uniform int8 * uniform dest,
	const uniform int8 * uniform src, const uniform int n) {
	foreach (i = 0 ... n) dest[i] = src[i];
}

// Copies only pages containing the input pattern
export void bookfilter_lch(const struct String * uniform page,
	const uniform int num_pages, const uniform struct String& pattern,
	struct String * uniform output) {

	foreach (i = 0 ... num_pages) {

		bool match = false;
		str_match_par(page[i].data, page[i].length, pattern, match);

		// Launching mem_cpy_simd once per thread that matched the pattern
		for (uniform int j = 0; j < programCount; ++j) {

			if ((j == programIndex) && (match != false)) {
				uniform int p = extract(i, j);
				launch task_mem_cpy_simd(output[p].data, page[p].data,
					page[p].length);
				output[p].length = page[p].length;
			}

		}

		sync;

	}

}
