SOURCEDIR = .

include ../../Makefile.common

run: compile
	@echo "====RUNNING BOTS/KNAPSACK===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;

run-parallel:
	@echo "\nRunning manually annotated:"
	@echo "\nSmall input:\n"
	time ./$(MI_OUT) input/knapsack-032.input;
	@echo "\nMedium input:\n"
	time ./$(MI_OUT) input/knapsack-064.input;
	@echo "\nBig input:\n"
	time ./$(MI_OUT) input/knapsack-128.input;

	@echo "\nRunning automatically annotated:"
	@echo "\nSmall input:\n"
	time ./$(AI_OUT) input/knapsack-032.input;
	@echo "\nMedium input:\n"
	time ./$(AI_OUT) input/knapsack-064.input;
	@echo "\nBig input:\n"
	time ./$(AI_OUT) input/knapsack-128.input;

run-serial:	
	@echo "\nRunning serial:"
	@echo "\nSmall input:\n"
	time ./$(SER_OUT) input/knapsack-032.input;
	@echo "\nMedium input:\n"
	time ./$(SER_OUT) input/knapsack-064.input;
	@echo "\nBig input:\n"
	time ./$(SER_OUT) input/knapsack-128.input;