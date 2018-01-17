include ../../Makefile.common

SOURCEDIR = .

TEST_INPUT = 10
SMALL_INPUT = 1000
MEDIUM_INPUT = 10000
BIG_INPUT = 100000

run: compile
	@echo "====RUNNING BOTS/LIST===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;