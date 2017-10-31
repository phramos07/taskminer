# include <vector>
# include <fstream>
# include <iostream>

# include "string_ispc.h"
# include "../../timing.h"
# include "ldp/common.hpp"

# define MAX_LENGTH (1 << 13)


int main (int argc, char *argv[]) {

	double elapsed(0);

	ispc::String text;
	ispc::String pattern;

	int num_matches(0);
	std::vector<int> matches(MAX_LENGTH, 0);

	// Input pattern
	std::string word(MAX_LENGTH, '\0');
	std::cin.getline(&word[0], MAX_LENGTH);

	pattern.data = (int8_t *) word.data();
	pattern.length = std::cin.gcount() - 1;

	// Input data
	std::string buffer(MAX_LENGTH, '\0');
	std::cin.read(&buffer[0], MAX_LENGTH);

	while (std::cin.gcount()) {

		text.data = (int8_t *) buffer.data();
		text.length = std::cin.gcount();
		buffer[text.length] = 0;

		// Looking for pattern in input string
		reset_and_start_timer();
		ispc::String_match_par(text, pattern, matches.data(), num_matches);
		elapsed += get_elapsed_mcycles();

		// Copying the last (|pattern| - 1) characters to the beginning
		// of the buffer before reading the next chunk of data
		int len(pattern.length - 1);
		buffer.replace(0, len, &buffer[text.length - 1 - len]);
		std::cin.read(&buffer[len], MAX_LENGTH);

	}

	print_runtime(elapsed);
	return 0;

}
