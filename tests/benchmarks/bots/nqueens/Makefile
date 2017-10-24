SOURCEDIR = .

include ../../Makefile.common

SMALL_INPUT = 13
MEDIUM_INPUT = 14
LARGE_INPUT = 15
HUGE_INPUT = 16

run: compile
	@echo "====RUNNING BOTS/NQUEENS===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;