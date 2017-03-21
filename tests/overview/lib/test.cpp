#include "Graph.hpp"

int main(int argc, char const *argv[])
{
	Graph * G = new Graph(10);
	// G->readFromInput(std::cin);
	G->printToDot(std::cout);
	
	return 0;
}