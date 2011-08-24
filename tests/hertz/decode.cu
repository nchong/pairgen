/*
 * Decode a LAMMPS neighbor list
 */
#ifndef DECODE_H
#define DECODE_H

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include "cuda_common.h"

//#define DEBUG //< sanity check neighbor list construction

/*
 * For each particle (tid) determine which page of the CPU neighbor list
 * contains the neighbor list for this particle.
 * TODO: further unwrap p into decomposition?
 */
__global__ void decode_neighlist_p1(
  //inputs
  int nparticles,
  int **firstneigh, //nb: contains cpu pointers: do not dereference!
  int maxpage,
  int **pages,      //nb: contains cpu pointers: do not dereference!
  int pgsize,
  //outputs
  int *pageidx
) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < nparticles) {
    int *myfirstneigh = firstneigh[tid];
    int mypage = -1;
    for (int p=0; p<maxpage; p++) {
      if ( (pages[p] <= myfirstneigh) &&
                       (myfirstneigh < (pages[p]+pgsize)) ) {
        mypage = p;
      }
    }
    pageidx[tid] = mypage;
  }
}

class GpuNeighList {
  public:
    int nparticles;
    int maxpage;
    int pgsize;
    //device neighbor list structures
    int  *d_numneigh;
    int **d_firstneigh;
    int **d_pages;
    int  *d_pageidx;
    int  *d_offset;
    int  *d_neighidx;
    //sizes of above
    size_t d_numneigh_size;
    size_t d_firstneigh_size;
    size_t d_pages_size;
    size_t d_pageidx_size;
    size_t d_offset_size;
    size_t d_neighidx_size;

  GpuNeighList(int _nparticles, int _maxpage, int _pgsize) :
    nparticles(_nparticles), maxpage(_maxpage), pgsize(_pgsize) {
    d_numneigh_size = nparticles * sizeof(int);
    d_firstneigh_size = nparticles * sizeof(int *);
    d_pages_size = maxpage * sizeof(int *);
    d_pageidx_size = d_numneigh_size;
    d_offset_size = d_numneigh_size;
    d_neighidx_size = maxpage * pgsize * sizeof(int);

    ASSERT_NO_CUDA_ERROR(
      cudaMalloc((void **)&d_numneigh, d_numneigh_size));
    ASSERT_NO_CUDA_ERROR(
      cudaMalloc((void **)&d_firstneigh, d_firstneigh_size));
    ASSERT_NO_CUDA_ERROR(
      cudaMalloc((void **)&d_pages, d_pages_size));
    ASSERT_NO_CUDA_ERROR(
      cudaMalloc((void **)&d_pageidx, d_pageidx_size));
    ASSERT_NO_CUDA_ERROR(
      cudaMalloc((void **)&d_offset, d_offset_size));
    ASSERT_NO_CUDA_ERROR(
      cudaMalloc((void **)&d_neighidx, d_neighidx_size));
  }

  ~GpuNeighList() {
    cudaFree(d_numneigh);
    cudaFree(d_firstneigh);
    cudaFree(d_pages);
    cudaFree(d_pageidx);
    cudaFree(d_offset);
    cudaFree(d_neighidx);
  }

  double fill_ratio() {
    //thrust versions of raw device pointers
    static thrust::device_ptr<int > thrust_numneigh(d_numneigh);
    int sum_numneigh = thrust::reduce(thrust_numneigh, thrust_numneigh + nparticles);
    int nslot = maxpage * pgsize;
    return ((double)sum_numneigh / (double)(nslot));
  }

  template <class T>
  inline void load_pages(T *d_ptr, T **h_ptr, int arity=1) {
    for (int p=0; p<maxpage; p++) {
      ASSERT_NO_CUDA_ERROR(
        cudaMemcpy(&(d_ptr[p*pgsize*arity]), h_ptr[p], pgsize*sizeof(T)*arity, cudaMemcpyHostToDevice));
    }
  }

  template <class T>
  inline void unload_pages(T *d_ptr, T **h_ptr, int arity=1) {
    for (int p=0; p<maxpage; p++) {
      ASSERT_NO_CUDA_ERROR(
        cudaMemcpy(h_ptr[p], &(d_ptr[p*pgsize*arity]), pgsize*sizeof(T)*arity, cudaMemcpyDeviceToHost));
    }
  }

  void reload(int *numneigh, int **firstneigh, int **pages) {
    //thrust versions of raw device pointers
    static thrust::device_ptr<int > thrust_numneigh(d_numneigh);
    static thrust::device_ptr<int>  thrust_pageidx(d_pageidx);
    static thrust::device_ptr<int>  thrust_offset(d_offset);

    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_firstneigh, firstneigh, d_firstneigh_size, cudaMemcpyHostToDevice));
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_pages, pages, d_pages_size, cudaMemcpyHostToDevice));

#ifdef KERNEL_PRINT
    cudaPrintfInit();
#endif

    const int blockSize = 128;
    dim3 gridSize((nparticles/ blockSize)+1);
    decode_neighlist_p1<<<gridSize, blockSize>>>(
        nparticles,
        d_firstneigh,
        maxpage,
        d_pages,
        pgsize,
        d_pageidx
        );

#ifdef KERNEL_PRINT
    cudaThreadSynchronize();
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
#endif

#ifdef DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Post decode_neighlist_p1 kernel error: %s.\n", cudaGetErrorString(err));
      exit(1);
    }

    // test that each assigned pageidx is sane
    for (int i=0; i<nparticles; i++) {
      assert(0 <= thrust_pageidx[i]);
      assert(     thrust_pageidx[i] < maxpage);
      if (i > 0) {
        assert(thrust_pageidx[i] == thrust_pageidx[i-1] ||
               thrust_pageidx[i] == thrust_pageidx[i-1]+1);
      }
    }
#endif

    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_numneigh, numneigh, d_numneigh_size, cudaMemcpyHostToDevice));
    thrust::exclusive_scan_by_key(
      thrust_pageidx,              // ] keys
      thrust_pageidx + nparticles, // ] 
      thrust_numneigh,             //vals
      thrust_offset);              //output

    load_pages<int>(d_neighidx, pages);

#ifdef DEBUG
    // test equality
    static thrust::device_ptr<int>  thrust_neighidx(d_neighidx);
    for (int i=0; i<nparticles; i++) {
      for (int j=0; j<numneigh[i]; j++) {
        int expected = firstneigh[i][j];
        int mypage = thrust_pageidx[i];
        int myoffset = thrust_offset[i];
        int actual = thrust_neighidx[(mypage*pgsize)+myoffset+j];
        assert(expected == actual);
      }
    }
#endif
  }
};

#endif
