struct String {

	uniform int length;
	int8 * uniform data;

};


// Matches the pattern with the input text
void pattern_match(uniform String& text, uniform String& pattern,
	uniform int offset) {

	varying bool match = true;
	foreach (i = 1 ... pattern.length) {
		if (pattern.data[i] != text.data[i + offset]) match &= false;
	}

	if (all(match)) print("match => %\n", offset);

}

// Matches the pattern with the input text
task void task_pattern_match(uniform String& text, uniform String& pattern,
	uniform int offset) {

	varying bool match = true;
	foreach (i = 1 ... pattern.length) {
		if (pattern.data[i] != text.data[i + offset]) match &= false;
	}

	if (all(match)) print("match => %\n", offset);

}

// Matches the pattern with the input text
void opt_spmd_pattern_match(uniform String& text, uniform String& pattern,
	varying int offset) {

	// For each worker "i" in the gang
	for (uniform int i = 0; i < programCount; ++i) {

		// If it was active at the callsite of this function
		if (programIndex == i) {

			// "Broadcast" parameters to all workers
			uniform int local_offset = extract(offset, i);

			unmasked {

				// Invoke pattern_match with all threads
				pattern_match(text, pattern, local_offset);

			}

		}

	}

}

// Matches the pattern with the input text
void spmd_pattern_match(uniform String& text, uniform String& pattern,
	varying int offset) {

	uniform int local_offset = -1;

	// For each worker "i" in the gang
	for (uniform int i = 0; i < programCount; ++i) {

		// If it was active at the callsite of this function
		if (programIndex == i) {

			// "Broadcast" parameters to all workers
			local_offset = extract(offset, i);

		}

		if (local_offset != -1) {

			// Invoke pattern_match with all threads
			pattern_match(text, pattern, local_offset);

		}

		local_offset = -1;

	}

}

// Finds the given pattern along the input text
export void String_match_ldp(uniform String& text, uniform String& pattern,
	int * uniform matches, uniform int& num_matches) {

	for (uniform int i = 0; i < text.length; i += programCount) {

		varying int pos = i + programIndex;
		if (text.data[pos] == pattern.data[0]
			&& (pos + pattern.length) <= text.length) {

			ldp pattern_match(text, pattern, pos);

		}

	}

}

export void String_match_lch(uniform String& text, uniform String& pattern,
	int * uniform matches, uniform int& num_matches) {

	for (uniform int i = 0; i < text.length; i += programCount) {

		for (uniform int j = 0; j < programCount; ++j) {
			uniform int pos = i + j;
			if (text.data[pos] == pattern.data[0]
				&& (pos + pattern.length) <= text.length) {
				launch task_pattern_match(text, pattern, pos);
			}
		}

		sync;

	}

}

export void String_match_par(uniform String& text, uniform String& pattern,
		int * uniform matches, uniform int& num_matches) {

	for (uniform int i = 0; i < text.length; i += programCount) {

		varying int pos = i + programIndex;
		varying int j = 0;

		while (text.data[pos] == pattern.data[j]
				&& (pos + pattern.length) <= text.length
				&& j < pattern.length) {
			j++;
			pos++;
		}

		// Check if found
		if (j == pattern.length) {
			print("match => %\n", (pos - j));
		}

	}

}
