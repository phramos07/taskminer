include ../../Makefile.common

SOURCEDIR = .

TEST_INPUT = 10 80 pdf
SMALL_INPUT = 10 500000 pdf
MEDIUM_INPUT = 30 30000000 pdf
BIG_INPUT = 50 100000000 pdf

SERIAL_FILES = bookfilter.c

run: compile
	@echo "====RUNNING SAMPLE/BOOKFILTER===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;

