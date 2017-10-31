include ../../Makefile.common

SOURCEDIR = .

TEST_INPUT = input/small
SMALL_INPUT = input/small
MEDIUM_INPUT = input/medium
BIG_INPUT = 

run: compile
	@echo "====RUNNING BOTS/BOOKFILTER===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;