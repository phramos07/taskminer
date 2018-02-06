#include "poisson.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define min(a, b) ((a < b) ? a : b)
#define max(a, b) ((a > b) ? a : b)

/* #pragma omp task/taskwait version of SWEEP. */
void sweep(int nx, int ny, double dx, double dy, double *f_, int itold,
           int itnew, double *u_, double *unew_, int block_size) {
  int it;
  int block_x, block_y;

  if (block_size == 0)
    block_size = nx;

  int max_blocks_x = (nx / block_size);
  int max_blocks_y = (ny / block_size);

  for (it = itold + 1; it <= itnew; it++) {
    // Save the current estimate.
    for (block_x = 0; block_x < max_blocks_x; block_x++) {
      for (block_y = 0; block_y < max_blocks_y; block_y++) {
        copy_block(nx, ny, block_x, block_y, u_, unew_, block_size);
      }
    }

    // Compute a new estimate.
    for (block_x = 0; block_x < max_blocks_x; block_x++) {
      for (block_y = 0; block_y < max_blocks_y; block_y++) {
        compute_estimate(block_x, block_y, u_, unew_, f_, dx, dy, nx, ny,
                         block_size);
      }
    }
  }
}

int comp(const void *elem1, const void *elem2) {
  double f = *((double *)elem1);
  double s = *((double *)elem2);
  if (f > s)
    return 1;
  if (f < s)
    return -1;
  return 0;
}

int main(int argc, char *argv[]) {
  // int num_threads = 1;
  user_parameters params;
  memset(&params, 0, sizeof(params));

  /* default value */
  params.niter = 1;
  params.matrix_size = atoi(argv[1]);
  params.blocksize = atoi(argv[2]);

  // warmup
  run(&params);

  //     double mean = 0.0;
  //     double meansqr = 0.0;
  //     double min_ = DBL_MAX;
  //     double max_ = -1;
  //     double* all_times = (double*)malloc(sizeof(double) * params.niter);

  //     for (int i=0; i<params.niter; ++i)
  //     {
  //       double cur_time = run(&params);
  //       all_times[i] = cur_time;
  //       mean += cur_time;
  //       min_ = min(min_, cur_time);
  //       max_ = max(max_, cur_time);
  //       meansqr += cur_time * cur_time;
  //       }
  //     mean /= params.niter;
  //     meansqr /= params.niter;
  //     double stddev = sqrt(meansqr - mean * mean);

  //     qsort(all_times, params.niter, sizeof(double), comp);
  //     double median = all_times[params.niter / 2];

  //     free(all_times);

  //     printf("Program : %s\n", argv[0]);
  // #ifdef MSIZE
  //     printf("Size : %d\n", params.matrix_size);
  // #endif
  // #ifdef SMSIZE
  //     printf("Submatrix size : %d\n", params.submatrix_size);
  // #endif
  // #ifdef BSIZE
  //     printf("Blocksize : %d\n", params.blocksize);
  // #endif
  // #ifdef IBSIZE
  //     printf("Internal Blocksize : %d\n", params.iblocksize);
  // #endif
  // #ifdef TITER
  //     printf("Iteration time : %d\n", params.titer);
  // #endif
  //     printf("Iterations : %d\n", params.niter);
  // #ifdef CUTOFF_SIZE
  //     printf("Cutoff Size : %d\n", params.cutoff_size);
  // #endif
  // #ifdef CUTOFF_DEPTH
  //     printf("Cutoff depth : %d\n", params.cutoff_depth);
  // #endif
  //     printf("Threads : %d\n", num_threads);
  // #ifdef GFLOPS
  //     printf("Gflops:: ");
  // #else
  //     printf("Time(sec):: ");
  // #endif
  //     printf("avg : %lf :: std : %lf :: min : %lf :: max : %lf :: median :
  //     %lf\n",
  //            mean, stddev, min_, max_, median);
  // if(params.check)
  printf("Check : %s\n",
         (params.succeed)
             ? ((params.succeed > 1) ? "not implemented" : "success")
             : "fail");
  if (params.string2display != 0)
    printf("%s", params.string2display);
  printf("\n");

  return 0;
}
