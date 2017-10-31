# include <iomanip>
# include <iostream>


// Prints out the runtime of the program
void print_runtime(double elapsed) {
	std::cerr << std::fixed, std::cerr.precision(3);
	std::cerr << elapsed << " million cycles" << std::endl;
}
