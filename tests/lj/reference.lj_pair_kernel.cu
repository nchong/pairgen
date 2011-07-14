#ifndef LJ_PAIR_KERNEL_H
#define LJ_PAIR_KERNEL_H

#include "lj_constants.h"

#ifdef DEBUG
#include "cuPrintf.cu"
#define WATCHI 1
#define DEBUG_PRINT_INPUTS(printi) {   \
  if (i == printi || j == printi) {    \
    cuPrintf("i = %d; j= %d\n", i, j); \
    cuPrintf("xi[0] = %.16f\n", xi[0]); \
    cuPrintf("xi[1] = %.16f\n", xi[1]); \
    cuPrintf("xi[2] = %.16f\n", xi[2]); \
    cuPrintf("xj[0] = %.16f\n", xi[0]); \
    cuPrintf("xj[1] = %.16f\n", xi[1]); \
    cuPrintf("xj[2] = %.16f\n", xi[2]); \
    cuPrintf("radiusi = %.16f\n", radiusi); \
    cuPrintf("radiusj = %.16f\n", radiusi); \
    cuPrintf("typei = %d\n", typei); \
    cuPrintf("typej = %d\n", typei); \
    cuPrintf("forcei_delta[0] = %.16f\n", forcei_delta[0]); \
    cuPrintf("forcei_delta[1] = %.16f\n", forcei_delta[1]); \
    cuPrintf("forcei_delta[2] = %.16f\n", forcei_delta[2]); \
  }                                   \
} while(0);
#define DEBUG_PRINT_OUTPUTS(printi) { \
  if (i == printi || j == printi) {   \
    cuPrintf("new forcei_delta[0] = %.16f\n", forcei_delta[0]); \
    cuPrintf("new forcei_delta[1] = %.16f\n", forcei_delta[1]); \
    cuPrintf("new forcei_delta[2] = %.16f\n", forcei_delta[2]); \
  }                                   \
} while(0);
#endif

/*
 * Pairwise interaction of particles i and j
 * Constants:
 * 	d_lj1 
 * 	d_lj2 
 * 	d_cutsq 
 * Read-Only:
 * 	xi, xj	-- position
 * 	radiusi, radiusj
 * 	typei, typej
 * Update:
 * 	forcei_delta
 */
__device__ void lj_pair_kernel(
#ifdef DEBUG
    int i, int j,
#endif
    double xi[3], 
    double xj[3], 
    double radiusi, 
    double radiusj, 
    int typei, 
    int typej, 
    double forcei_delta[3]
    ) {
  // del is the vector from j to i
  double delx = xi[0] - xj[0];
  double dely = xi[1] - xj[1];
  double delz = xi[2] - xj[2];
  double rsq = delx*delx + dely*dely + delz*delz;
  if (rsq < d_cutsq) {
    double r2inv = 1.0/rsq;
    double r6inv = r2inv*r2inv*r2inv;
    double forcelj = r6inv * (d_lj1*r6inv - d_lj2);
    double fpair = forcelj*r2inv;

    forcei_delta[0] += delx*fpair;
    forcei_delta[1] += dely*fpair;
    forcei_delta[2] += delz*fpair;
  }
}

#endif
