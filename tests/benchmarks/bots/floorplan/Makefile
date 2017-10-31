include ../../Makefile.common

SOURCEDIR = .

SMALL_INPUT = input/small
MEDIUM_INPUT = input/medium
BIG_INPUT = input/big

run: compile
	@echo "====RUNNING BOTS/FLOORPLAN===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;

