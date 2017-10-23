SOURCEDIR = .

include ../../Makefile.common

run: compile
	@echo "====RUNNING BOTS/HEALTH===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;

run-parallel:
	@echo "\nRunning manually annotated:"
	@echo "\nSmall input:\n"
	time ./$(MI_OUT) input/small.input;
	@echo "\nMedium input:\n"
	time ./$(MI_OUT) input/medium.input;
	@echo "\nBig input:\n"
	time ./$(MI_OUT) input/large.input;

	@echo "\nRunning automatically annotated:"
	@echo "\nSmall input:\n"
	time ./$(AI_OUT) input/small.input;
	@echo "\nMedium input:\n"
	time ./$(AI_OUT) input/medium.input;
	@echo "\nBig input:\n"
	time ./$(AI_OUT) input/large.input;

run-serial:	
	@echo "\nRunning serial:"
	@echo "\nSmall input:\n"
	time ./$(SER_OUT) input/small.input;
	@echo "\nMedium input:\n"
	time ./$(SER_OUT) input/medium.input;
	@echo "\nBig input:\n"
	time ./$(SER_OUT) input/large.input;