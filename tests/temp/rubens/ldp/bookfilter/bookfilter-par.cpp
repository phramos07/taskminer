# include <vector>
# include <fstream>
# include <iostream>
# include <algorithm>

# include "../../timing.h"
# include "ldp/common.hpp"
# include "bookfilter_ispc.h"

# define MAX_LENGTH (1 << 13)


int main (int argc, char *argv[]) {

	double elapsed(0);

	size_t num_lines;
	std::cin >> num_lines, std::cin.ignore(256, '\n');

	ispc::String *page = new ispc::String[num_lines];
	ispc::String *output = new ispc::String[num_lines];
	ispc::String *pattern = new ispc::String;

	std::string input_pattern;
	std::getline(std::cin, input_pattern);

	pattern->length = input_pattern.size();
	pattern->data = (int8_t *) input_pattern.data();

	std::string line;
	size_t count = 0;
	while (std::getline(std::cin, line)) {

		uint32_t len = line.length();

		page[count].length = len;
		page[count].data = new int8_t[len + 1];
		std::copy_n((int8_t *) line.data(), len, page[count].data);

		output[count].length = 0;
		output[count].data = new int8_t[len + 1];

		++count;

	}

	reset_and_start_timer();
	ispc::bookfilter_par(page, count, *pattern, output);
	elapsed += get_elapsed_mcycles();

	for (int i = 0; i < count; ++i) {

		/*
		std::cout << "[";
		for (int j = 0; j < page[i].length; ++j) {
			std::cout << (char) page[i].data[j];
		}
		std::cout << "] : [";
		if (output[i].length == 0) std::cout << "--";
		for (int j = 0; j < output[i].length; ++j) {
			std::cout << (char) output[i].data[j];
		}
		std::cout << "]\n";
		*/

		delete[] page[i].data;
		delete[] output[i].data;

	}

	delete pattern;

	delete[] page;
	delete[] output;

	print_runtime(elapsed);
	return 0;

}
