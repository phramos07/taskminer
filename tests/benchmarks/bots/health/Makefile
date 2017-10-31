SOURCEDIR = .

include ../../Makefile.common

TEST_INPUT = input/small.input
SMALL_INPUT = input/small.input
MEDIUM_INPUT = input/medium.input
BIG_INPUT = input/large.input

run: compile
	@echo "====RUNNING BOTS/HEALTH===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;