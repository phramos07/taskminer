include ../../Makefile.common

SOURCEDIR = .

TEST_INPUT = 15
SMALL_INPUT = 45

run: compile
	@echo "====RUNNING BOTS/FIB===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;

run-ai:
	@echo "\nRunning automatically annotated:"
	@echo "\nSmall input:\n"
	time ./$(AI_OUT) $(SMALL_INPUT);

run-mi:
	@echo "\nRunning manually annotated:"
	@echo "\nSmall input:\n"
	time ./$(MI_OUT) $(SMALL_INPUT);

run-serial:	
	@echo "\nRunning serial:"
	@echo "\nSmall input:\n"
	time ./$(SER_OUT) $(SMALL_INPUT);