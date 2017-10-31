SOURCEDIR = .

include ../../Makefile.common

TINY_INPUT = input/tiny.input
TEST_INPUT = input/test.input
SMALL_INPUT = input/small.input
MEDIUM_INPUT = input/medium.input
BIG_INPUT = input/large.input

export OMP_STACKSIZE=150M

# [test]
# inputs=test.input
# ompenv=OMP_STACKSIZE="2M"

# [small]
# inputs=small.input
# ompenv=OMP_STACKSIZE="16M"

# [medium]
# inputs=medium.input
# ompenv=OMP_STACKSIZE="55M"

# [large]
# inputs=large.input
# ompenv=OMP_STACKSIZE="150M"

run: compile
	@echo "====RUNNING BOTS/UTS===="
	$(MAKE) -i run-parallel;
	$(MAKE) -i run-serial;

compile-serial:
	$(CC_OMP) $(CFLAGS) uts_SER.c brg_sha1.c $(LFLAGS) -o uts_SER.c$(SER_SUFFIX).out;

compile-parallel:	
	$(CC_OMP) $(OMP_FLAGS) $(CFLAGS) uts_MI.c brg_sha1.c $(LFLAGS) -o uts_MI.c$(MI_SUFFIX).out;
	$(CC_OMP) $(OMP_FLAGS) $(CFLAGS) uts_AI.c brg_sha1.c $(LFLAGS) -o uts_AI.c$(AI_SUFFIX).out;