include ../../Makefile.common

SOURCEDIR = .

TEST_INPUT = < input/test_input
SMALL_INPUT = < input/small_input
MEDIUM_INPUT = < input/medium_input
BIG_INPUT = < input/big_input

run: compile
	@echo "====RUNNING TOYS/QUADTREE===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;