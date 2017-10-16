# include <string>
# include <vector>

# include "../../timing.h"
# include "ldp/common.hpp"

# define MAX_LENGTH (1 << 13)


// Precomputes the matching table to the given pattern
void preKMP(const std::string& pattern, std::vector<int>& table) {

	table.resize(0);
	table.resize(pattern.size());

	int pos(0);
	table[pos] = 0;

	// Computing table[i] for the entire pattern
	int i(1), length(pattern.size());
	while (i < length) {

		if (pattern[i] == pattern[pos]) {
			++pos, table[i] = pos, ++i;
		} else if (pos != 0) {
			pos = table[pos - 1];
		} else table[i] = 0, ++i;

	}

}

// Finds occurrences of pattern in the target text
void KMP(const std::string& text, const std::string& pattern,
	const std::vector<int>& table) {

	int m(pattern.size()), n(text.size());

	int i(0), j(0);
	while (i < n) {

		if (pattern[j] == text[i]) ++i, ++j;

		if (j == m) {

			std::cout << "match => " << (i - j) << "\n";
			j = table[j - 1];

		} else if (i < n and pattern[j] != text[i]) {

			// Skipping characters from table[0 .. table[j - 1]],
			// as they have already matched
			if (j != 0) j = table[j-1];
			else ++i;

		}

	}

}


int main(int argc, char *argv[]) {

	double elapsed(0);

	// Input pattern
	std::string pattern(MAX_LENGTH, '\0');
	std::cin.getline(&pattern[0], MAX_LENGTH);
	pattern.resize(std::cin.gcount() - 1);

	std::vector<int> table;

	reset_and_start_timer();
	preKMP(pattern, table);
	elapsed += get_elapsed_mcycles();

	// Input data
	std::string text(MAX_LENGTH, '\0');
	std::cin.read(&text[0], MAX_LENGTH);

	while (std::cin.gcount()) {

		text.resize(std::cin.gcount());

		// Looking for pattern in input string
		reset_and_start_timer();
		KMP(text, pattern, table);
		elapsed += get_elapsed_mcycles();

		// Copying the last |pattern| characters to the beginning
		// of the buffer before reading the next chunk of data
		int len(pattern.size());
		text.replace(0, len, &text[text.length() - len]);
		std::cin.read(&text[len], MAX_LENGTH);

	}

	print_runtime(elapsed);
	return 0;

}
