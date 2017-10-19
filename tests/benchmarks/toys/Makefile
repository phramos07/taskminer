include ../Makefile.common

BELLMANFORD = bellmanford
BOOKFILTER = bookfilter
BORUVKA = boruvka
DFS = dfs
FIBONACCI = fibonacci
LIST = list
REGIONS = regions

SUBDIRS = $(BELLMANFORD) $(BOOKFILTER) $(BORUVKA) $(DFS) $(FIBONACCI) $(LIST) $(REGIONS)

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

