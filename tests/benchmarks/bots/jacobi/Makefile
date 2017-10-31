SOURCEDIR = .

include ../../Makefile.common

TEST_INPUT = 1024 16
SMALL_INPUT = 4096 64
MEDIUM_INPUT = 8192 128
BIG_INPUT = 16384 256

export OMP_STACKSIZE=150M

run: compile
	@echo "====RUNNING KASTORS/JACOBI===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;

compile-serial:
	$(CC_OMP) $(CFLAGS) $(SERIAL_FILES) poisson.c $(LFLAGS) -o jacobi_SER.c$(SER_SUFFIX).out;

compile-parallel:	
	$(CC_OMP) $(OMP_FLAGS) $(CFLAGS) $(MI_FILES) poisson.c $(LFLAGS) -o jacobi_MI.c$(MI_SUFFIX).out;
	$(CC_OMP) $(OMP_FLAGS) $(CFLAGS) $(AI_FILES) poisson.c $(LFLAGS) -o jacobi_AI.c$(AI_SUFFIX).out;