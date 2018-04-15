SOURCEDIR = .

include ../../Makefile.common

TEST_INPUT = input/knapsack-032.input
SMALL_INPUT = input/knapsack-036.input
MEDIUM_INPUT = input/knapsack-040.input
BIG_INPUT  = input/knapsack-044.input

run: compile
	@echo "====RUNNING BOTS/KNAPSACK===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;
