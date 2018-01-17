include ../../Makefile.common

SOURCEDIR = .

TEST_INPUT = 10
SMALL_INPUT = 5000
MEDIUM_INPUT = 15000
BIG_INPUT = 30000

run: compile
	@echo "====RUNNING TOYS/DFS===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;