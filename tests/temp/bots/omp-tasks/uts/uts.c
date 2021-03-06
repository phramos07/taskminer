/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/**********************************************************************************************/
/*
 * Copyright (c) 2007 The Unbalanced Tree Search (UTS) Project Team:
 * -----------------------------------------------------------------
 *  
 *  This file is part of the unbalanced tree search benchmark.  This
 *  project is licensed under the MIT Open Source license.  See the LICENSE
 *  file for copyright and licensing information.
 *
 *  UTS is a collaborative project between researchers at the University of
 *  Maryland, the University of North Carolina at Chapel Hill, and the Ohio
 *  State University.
 *
 * University of Maryland:
 *   Chau-Wen Tseng(1)  <tseng at cs.umd.edu>
 *
 * University of North Carolina, Chapel Hill:
 *   Jun Huan         <huan,
 *   Jinze Liu         liu,
 *   Stephen Olivier   olivier,
 *   Jan Prins*        prins at cs.umd.edu>
 * 
 * The Ohio State University:
 *   James Dinan      <dinan,
 *   Gerald Sabin      sabin,
 *   P. Sadayappan*    saday at cse.ohio-state.edu>
 *
 * Supercomputing Research Center
 *   D. Pryor
 *
 * (1) - indicates project PI
 *
 * UTS Recursive Depth-First Search (DFS) version developed by James Dinan
 *
 * Adapted for OpenMP 3.0 Task-based version by Stephen Olivier
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

#include "app-desc.h"
#include "bots.h"
#include "uts.h"

/***********************************************************
 *  Global state                                           *
 ***********************************************************/
unsigned long long nLeaves = 0;
int maxTreeDepth = 0;
/***********************************************************
 * Tree generation strategy is controlled via various      *
 * parameters set from the command line.  The parameters   *
 * and their default values are given below.               *
 * Trees are generated using a Galton-Watson process, in   *
 * which the branching factor of each node is a random     *
 * variable.                                               *
 *                                                         *
 * The random variable follow a binomial distribution.     *
 ***********************************************************/
double b_0   = 4.0; // default branching factor at the root
int   rootId = 0;   // default seed for RNG state at root
/***********************************************************
 *  The branching factor at the root is specified by b_0.
 *  The branching factor below the root follows an 
 *     identical binomial distribution at all nodes.
 *  A node has m children with prob q, or no children with 
 *     prob (1-q).  The expected branching factor is q * m.
 *
 *  Default parameter values 
 ***********************************************************/
int    nonLeafBF   = 4;            // m
double nonLeafProb = 15.0 / 64.0;  // q
/***********************************************************
 * compute granularity - number of rng evaluations per
 * tree node
 ***********************************************************/
int computeGranularity = 1;
/***********************************************************
 * expected results for execution
 ***********************************************************/
unsigned long long  exp_tree_size = 0;
int        exp_tree_depth = 0;
unsigned long long  exp_num_leaves = 0;
/***********************************************************
 *  FUNCTIONS                                              *
 ***********************************************************/

// Interpret 32 bit positive integer as value on [0,1)
double rng_toProb(int n)
{
  if (n < 0) {
    printf("*** toProb: rand n = %d out of range\n",n);
  }
  return ((n<0)? 0.0 : ((double) n)/2147483648.0);
}

void uts_initRoot(Node * root)
{
   root->height = 0;
   root->numChildren = -1;      // means not yet determined
   rng_init(root->state.state, rootId);

   printf("Root node at %p\n", root);
}


int uts_numChildren_bin(Node * parent)
{
  // distribution is identical everywhere below root
  int    v = rng_rand(parent->state.state);	
  double d = rng_toProb(v);

  return (d < nonLeafProb) ? nonLeafBF : 0;
}

int uts_numChildren(Node *parent)
{
  int numChildren = 0;

  /* Determine the number of children */
  if (parent->height == 0) numChildren = (int) floor(b_0);
  else numChildren = uts_numChildren_bin(parent);
  
  // limit number of children
  // only a BIN root can have more than MAXNUMCHILDREN
  if (parent->height == 0) {
    int rootBF = (int) ceil(b_0);
    if (numChildren > rootBF) {
      printf("*** Number of children of root truncated from %d to %d\n", numChildren, rootBF);
      numChildren = rootBF;
    }
  }
  else {
    if (numChildren > MAXNUMCHILDREN) {
      printf("*** Number of children truncated from %d to %d\n", numChildren, MAXNUMCHILDREN);
      numChildren = MAXNUMCHILDREN;
    }
  }

  return numChildren;
}


unsigned long long serTreeSearch(int depth, Node *parent, int numChildren)
{
  unsigned long long subtreesize = 1, partialCount[numChildren];
  Node n[numChildren];
  int i, j;

  // Recurse on the children
  for (i = 0; i < numChildren; i++) {
     n[i].height = parent->height + 1;
     // The following line is the work (one or more SHA-1 ops)
     for (j = 0; j < computeGranularity; j++) {
	rng_spawn(parent->state.state, n[i].state.state, i);
     }
     partialCount[i] = serTreeSearch(depth+1, &n[i], uts_numChildren(&n[i]));
  }

  // computing total size
  for (i = 0; i < numChildren; i++) subtreesize += partialCount[i];

  return subtreesize;
}

/***********************************************************
 * Recursive depth-first implementation                    *
 ***********************************************************/

unsigned long long parallel_uts ( Node *root )
{
   unsigned long long num_nodes = 0 ;
   root->numChildren = uts_numChildren(root);

   printf("Computing Unbalance Tree Search algorithm ");

        num_nodes = parTreeSearch( 0, root, root->numChildren );

   printf(" completed!");

   return num_nodes;
}

#if defined(IF_CUTOFF)

unsigned long long parTreeSearch(int depth, Node *parent, int numChildren)
{
  Node n[numChildren], *nodePtr;
  int i, j;
  unsigned long long subtreesize = 1, partialCount[numChildren];

  // Recurse on the children
  for (i = 0; i < numChildren; i++) {
     nodePtr = &n[i];

     nodePtr->height = parent->height + 1;

     // The following line is the work (one or more SHA-1 ops)
     for (j = 0; j < computeGranularity; j++) {
	rng_spawn(parent->state.state, nodePtr->state.state, i);
     }

     nodePtr->numChildren = uts_numChildren(nodePtr);

	partialCount[i] = parTreeSearch(depth+1, nodePtr, nodePtr->numChildren);
  }


  for (i = 0; i < numChildren; i++) {
     subtreesize += partialCount[i];
  }

  return subtreesize;
}

#elif defined(FINAL_CUTOFF)

unsigned long long parTreeSearch(int depth, Node *parent, int numChildren)
{
  Node n[numChildren], *nodePtr;
  int i, j;
  unsigned long long subtreesize = 1, partialCount[numChildren];

  // Recurse on the children
  for (i = 0; i < numChildren; i++) {
     nodePtr = &n[i];

     nodePtr->height = parent->height + 1;

     // The following line is the work (one or more SHA-1 ops)
     for (j = 0; j < computeGranularity; j++) {
	rng_spawn(parent->state.state, nodePtr->state.state, i);
     }

     nodePtr->numChildren = uts_numChildren(nodePtr);

	partialCount[i] = parTreeSearch(depth+1, nodePtr, nodePtr->numChildren);
  }


  for (i = 0; i < numChildren; i++) {
     subtreesize += partialCount[i];
  }

  return subtreesize;
}

#elif defined(MANUAL_CUTOFF)

unsigned long long parTreeSearch(int depth, Node *parent, int numChildren)
{
  if (depth >= bots_cutoff_value) {
    return serTreeSearch(depth, parent, numChildren);
  }

  Node n[numChildren], *nodePtr;
  int i, j;
  unsigned long long subtreesize = 1, partialCount[numChildren];

  // Recurse on the children
  for (i = 0; i < numChildren; i++) {
     nodePtr = &n[i];

     nodePtr->height = parent->height + 1;

     // The following line is the work (one or more SHA-1 ops)
     for (j = 0; j < computeGranularity; j++) {
	rng_spawn(parent->state.state, nodePtr->state.state, i);
     }

     nodePtr->numChildren = uts_numChildren(nodePtr);

	partialCount[i] = parTreeSearch(depth+1, nodePtr, nodePtr->numChildren);
  }


  for (i = 0; i < numChildren; i++) {
     subtreesize += partialCount[i];
  }

  return subtreesize;
}

#else

unsigned long long parTreeSearch(int depth, Node *parent, int numChildren) 
{
  Node n[numChildren], *nodePtr;
  int i, j;
  unsigned long long subtreesize = 1, partialCount[numChildren];

  // Recurse on the children
  for (i = 0; i < numChildren; i++) {
     nodePtr = &n[i];

     nodePtr->height = parent->height + 1;

     // The following line is the work (one or more SHA-1 ops)
     for (j = 0; j < computeGranularity; j++) {
        rng_spawn(parent->state.state, nodePtr->state.state, i);
     }

     nodePtr->numChildren = uts_numChildren(nodePtr);

        partialCount[i] = parTreeSearch(depth+1, nodePtr, nodePtr->numChildren);
  }


  for (i = 0; i < numChildren; i++) {
     subtreesize += partialCount[i];
  }
  
  return subtreesize;
}

#endif

void uts_read_file ( char *filename )
{
   FILE *fin;

   if ((fin = fopen(filename, "r")) == NULL) {
      printf("Could not open input file (%s)\n", filename);
      exit (-1);
   }
   fscanf(fin,"%lf %lf %d %d %d %llu %d %llu",
             &b_0,
             &nonLeafProb,
             &nonLeafBF,
             &rootId,
             &computeGranularity,
             &exp_tree_size,
             &exp_tree_depth,
             &exp_num_leaves
   );
   fclose(fin);

   computeGranularity = max(1,computeGranularity);

   // Printing input data
   printf("\n");
   printf("Root branching factor                = %f\n", b_0);
   printf("Root seed (0 <= 2^31)                = %d\n", rootId);
   printf("Probability of non-leaf node         = %f\n", nonLeafProb);
   printf("Number of children for non-leaf node = %d\n", nonLeafBF);
   printf("E(n)                                 = %f\n", (double) ( nonLeafProb * nonLeafBF ) );
   printf("E(s)                                 = %f\n", (double) ( 1.0 / (1.0 - nonLeafProb * nonLeafBF) ) );
   printf("Compute granularity                  = %d\n", computeGranularity);
   printf("Random number generator              = "); rng_showtype();
}

void uts_show_stats( void )
{
   int nPes = atoi(bots_resources);
   int chunkSize = 0;

   printf("\n");
   printf("Tree size                            = %llu\n", (unsigned long long)  bots_number_of_tasks );
   printf("Maximum tree depth                   = %d\n", maxTreeDepth );
   printf("Chunk size                           = %d\n", chunkSize );
   printf("Number of leaves                     = %llu (%.2f%%)\n", nLeaves, nLeaves/(float)bots_number_of_tasks*100.0 ); 
   printf("Number of PE's                       = %.4d threads\n", nPes );
   printf("Wallclock time                       = %.3f sec\n", bots_time_program );
   printf("Overall performance                  = %.0f nodes/sec\n", (bots_number_of_tasks / bots_time_program) );
   printf("Performance per PE                   = %.0f nodes/sec\n", (bots_number_of_tasks / bots_time_program / nPes) );
}

int uts_check_result ( void )
{
   int answer = BOTS_RESULT_SUCCESSFUL;

   if ( bots_number_of_tasks != exp_tree_size ) {
      answer = BOTS_RESULT_UNSUCCESSFUL;
      printf("Incorrect tree size result (%llu instead of %llu).\n", bots_number_of_tasks, exp_tree_size);
   }

   return answer;
}
