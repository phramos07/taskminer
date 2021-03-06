#include <omp.h>
#ifndef taskminerutils
#define taskminerutils
static int taskminer_depth_cutoff = 0;
#define DEPTH_CUTOFF 18
char cutoff_test = 0;
#endif
/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de
 * Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify */
/*  it under the terms of the GNU General Public License as published by */
/*  the Free Software Foundation; either version 2 of the License, or */
/*  (at your option) any later version. */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful, */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/*  GNU General Public License for more details. */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License */
/*  along with this program; if not, write to the Free Software */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
 * USA            */
/**********************************************************************************************/

/*
 * Original code from the Cilk project
 *
 * Copyright (c) 2000 Massachusetts Institute of Technology
 * Copyright (c) 2000 Matteo Frigo
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include "app-desc.h"
// #define CHECK_SOLUTION

int best_so_far;

int compare(struct item *a, struct item *b) {
  double c = ((double)a->value / a->weight) - ((double)b->value / b->weight);

  if (c > 0)
    return -1;
  if (c < 0)
    return 1;
  return 0;
}

int read_input(const char *filename, struct item *items, int *capacity,
               int *n) {
  int i;
  FILE *f;

  if (filename == NULL)
    filename = "\0";
  f = fopen(filename, "r");
  if (f == NULL) {
    fprintf(stderr, "open_input(\"%s\") failed\n", filename);
    return -1;
  }
  /* format of the input: #items capacity value1 weight1 ... */
  fscanf(f, "%d", n);
  fscanf(f, "%d", capacity);

  for (i = 0; i < *n; ++i)
    fscanf(f, "%d %d", &items[i].value, &items[i].weight);

  fclose(f);

  /* sort the items on decreasing order of value/weight */
  /* cilk2c is fascist in dealing with pointers, whence the ugly cast */
  qsort(items, *n, sizeof(struct item),
        (int (*)(const void *, const void *))compare);

  return 0;
}

/*
 * return the optimal solution for n items (first is e) and
 * capacity c. Value so far is v.
 */
void knapsack_par(struct item *e, int c, int n, int v, int *sol, int l) {
  taskminer_depth_cutoff++;
  int with, without, best;
  double ub;

  /* base case: full knapsack or no items */
  if (c < 0) {
    *sol = INT_MIN;
    return;
  }

  /* feasible solution, with value v */
  if (n == 0 || c == 0) {
    *sol = v;
    return;
  }

  ub = (double)v + c * e->value / e->weight;

  if (ub < best_so_far) {
    /* prune ! */
    *sol = INT_MIN;
    return;
  }
  /*
      * compute the best solution without the current item in the knapsack
      */
  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  #pragma omp task untied default(shared) depend(in:e[1]) depend(out:without) shared(without)
  knapsack_par(e + 1, c, n - 1, v, &without, l + 1);

  /* compute the best solution with the current item in the knapsack */
  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  #pragma omp task untied default(shared) depend(in:e[1]) depend(out:with) shared(with)
  knapsack_par(e + 1, c - e->weight, n - 1, v + e->value, &with, l + 1);
#pragma omp taskwait

  best = with > without ? with : without;

  /*
      * notice the race condition here. The program is still
      * correct, in the sense that the best solution so far
      * is at least best_so_far. Moreover best_so_far gets updated
      * when returning, so eventually it should get the right
      * value. The program is highly non-deterministic.
      */
  if (best > best_so_far)
    best_so_far = best;

  *sol = best;
taskminer_depth_cutoff--;
}
void knapsack_seq(struct item *e, int c, int n, int v, int *sol) {
  taskminer_depth_cutoff++;
  int with, without, best;
  double ub;

  /* base case: full knapsack or no items */
  if (c < 0) {
    *sol = INT_MIN;
    return;
  }

  /* feasible solution, with value v */
  if (n == 0 || c == 0) {
    *sol = v;
    return;
  }

  ub = (double)v + c * e->value / e->weight;

  if (ub < best_so_far) {
    /* prune ! */
    *sol = INT_MIN;
    return;
  }
  /*
      * compute the best solution without the current item in the knapsack
      */
  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  #pragma omp task untied default(shared) depend(in:e[1]) depend(out:without) shared(without)
  knapsack_seq(e + 1, c, n - 1, v, &without);

  /* compute the best solution with the current item in the knapsack */
  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  #pragma omp task untied default(shared) depend(in:e[1]) depend(out:with) shared(with)
  knapsack_seq(e + 1, c - e->weight, n - 1, v + e->value, &with);
#pragma omp taskwait

  best = with > without ? with : without;

  /*
      * notice the race condition here. The program is still
      * correct, in the sense that the best solution so far
      * is at least best_so_far. Moreover best_so_far gets updated
      * when returning, so eventually it should get the right
      * value. The program is highly non-deterministic.
      */
  if (best > best_so_far)
    best_so_far = best;

  *sol = best;
taskminer_depth_cutoff--;
}
void knapsack_main_par(struct item *e, int c, int n, int *sol) {
  best_so_far = INT_MIN;
  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  #pragma omp parallel
  #pragma omp single
  #pragma omp task untied default(shared)
  knapsack_par(e, c, n, 0, sol, 0);
  printf("Best value for parallel execution is %d\n\n", *sol);
}
void knapsack_main_seq(struct item *e, int c, int n, int *sol) {
  best_so_far = INT_MIN;
  cutoff_test = (taskminer_depth_cutoff < DEPTH_CUTOFF);
  #pragma omp parallel
  #pragma omp single
  #pragma omp task untied default(shared)
  knapsack_seq(e, c, n, 0, sol);

  printf("Best value for sequential execution is %d\n\n", *sol);
}

int knapsack_check(int sol_seq, int sol_par) {
  if (sol_seq == sol_par)
    return 1;
  else
    return -1;
}

int main(int argc, char const *argv[]) {
  struct item items[MAX_ITEMS];
  int n, capacity;
  int sol = 0;
  read_input(argv[1], items, &capacity, &n);

  // MAIN CALL
  knapsack_main_par(items, capacity, n, &sol);

#ifdef CHECK_SOLUTION
  int sol2 = 0;
  knapsack_main_seq(items, capacity, n, &sol2);

  if (knapsack_check(sol, sol2) != 1)
    printf("ERROR! Solution not correct!\n");
#endif

  return 0;
}

