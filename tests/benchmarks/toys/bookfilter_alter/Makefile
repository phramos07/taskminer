include ../../Makefile.common

SOURCEDIR = .

TEST_INPUT = 10 80 pdf
SMALL_INPUT = 10 500000 pdf
MEDIUM_INPUT = 30 30000000 pdf
BIG_INPUT = 100 10000000 pdf

run: compile
	@echo "====RUNNING TOYS/BOOKFILTER_ALTERED===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;