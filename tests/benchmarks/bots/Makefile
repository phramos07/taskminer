include ../Makefile.common

FFT = fft
FIB = fib
FLOORPLAN = floorplan
HEALTH = health
KNAPSACK = knapsack
NQUEENS = nqueens
SORT = sort
SPARSELU = sparselu
STRASSEN = strassen
UTS = uts

SUBDIRS = $(FFT) $(FLOORPLAN) $(HEALTH) $(KNAPSACK) $(NQUEENS) $(SORT) $(SPARSELU) $(STRASSEN) $(UTS)

compile:
	for DIR in $(SUBDIRS); do \
		$(MAKE) -C $$DIR compile-serial; \
		$(MAKE) -C $$DIR compile-parallel; \
	done

run: compile
	for DIR in $(SUBDIRS); do \
		$(MAKE) -C $$DIR run; \
	done

compile-serial:
	for DIR in $(SUBDIRS); do \
		$(MAKE) -C $$DIR compile-serial; \
	done

compile-parallel:
	for DIR in $(SUBDIRS); do \
		$(MAKE) -C $$DIR compile-parallel; \
	done

bc:
	for DIR in $(SUBDIRS); do \
		$(MAKE) -C $$DIR bc; \
	done

taskminer: bc
	for DIR in $(SUBDIRS); do \
		$(MAKE) -C $$DIR taskminer; \
	done

clean:
	for DIR in $(SUBDIRS); do \
		$(MAKE) -C $$DIR clean; \
	done

