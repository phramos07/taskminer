include ../../Makefile.common

SOURCEDIR = .

TEST_INPUT = 100
SMALL_INPUT = 1000
MEDIUM_INPUT = 10000
BIG_INPUT = 100000

SERIAL_FILES = priv_SER.c

run: compile
	@echo "====RUNNING SAMPLE/LIST===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;