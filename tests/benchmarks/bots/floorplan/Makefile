include ../../Makefile.common

SOURCEDIR = .

run: compile
	@echo "====RUNNING BOTS/FLOORPLAN===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;

run-parallel:
	@echo "\nRunning manually annotated:"
	@echo "\nSmall input:\n"
	time ./$(MI_OUT) input/small;
	@echo "\nMedium input:\n"
	time ./$(MI_OUT) input/medium;
	@echo "\nBig input:\n"
	time ./$(MI_OUT) input/big;

	@echo "\nRunning automatically annotated:"
	@echo "\nSmall input:\n"
	time ./$(AI_OUT) input/small;
	@echo "\nMedium input:\n"
	time ./$(AI_OUT) input/medium;
	@echo "\nBig input:\n"
	time ./$(AI_OUT) input/big;

run-serial:	
	@echo "\nRunning serial:"
	@echo "\nSmall input:\n"
	time ./$(SER_OUT) input/small;
	@echo "\nMedium input:\n"
	time ./$(SER_OUT) input/medium;
	@echo "\nBig input:\n"
	time ./$(SER_OUT) input/big;