include ../../Makefile.common

TEST_INPUT = 5
SMALL_INPUT = 2000
MEDIUM_INPUT = 10000
BIG_INPUT = 20000

SOURCEDIR = .

run: compile
	@echo "====RUNNING BOTS/BORUVKA===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;