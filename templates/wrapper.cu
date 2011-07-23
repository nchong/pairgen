#ifndef {{ name|upper }}_WRAPPER_H
#define {{ name|upper }}_WRAPPER_H

#include <cassert>
#include <cstdlib>
#include <cstdio>

#include "{{name}}_particle.h"
#include "{{name}}_tpa_compute_kernel.cu"
#include "{{name}}_bpa_compute_kernel.cu"
#include "decode.cu"

using namespace std;

#define ASSERT_NO_CUDA_ERROR( callReturningErrorstatus ) {     \
  cudaError_t err = callReturningErrorstatus;                  \
  if (err != cudaSuccess) {                                    \
    fprintf(stderr,                                            \
            "Cuda error (%s/%d) in file '%s' in line %i\n",    \
            cudaGetErrorString(err), err, __FILE__, __LINE__); \
    exit(1);                                                   \
  }                                                            \
} while(0);

/*
 * GPU datastructures passed to compute_kernel
 */
static struct particle d_particle_soa;
{% for p in params if not p.is_type('P', 'RO') -%}
  static {{ p.emit_pointer_to_declaration(name_prefix='d_') }};
{% endfor %}
static GpuNeighList *d_nl;

/*
 * Initialize GPU datastructures
 */
void {{name}}_init(int N, int maxpage, int pgsize,
  {% for p in params if p.is_type('-', 'RO') and not p.reload -%}
    {{ p.emit_pointer_to_declaration(name_prefix='h_') }}{{ ',' if not loop.last }}
  {% endfor -%}
) {
  {% for p in params if p.is_type('P', 'RO') -%}
    assert(d_particle_soa.{{ p.name }} == NULL);
  {% endfor %}
  d_nl = new GpuNeighList(N, maxpage, pgsize);
  {% for p in params if not p.is_type('P', 'RO') -%}
    assert({{ p.device_name() }} == NULL);
  {% endfor %}

  {% for p in params if p.is_type('P', 'RO') -%}
    ASSERT_NO_CUDA_ERROR(
      cudaMalloc((void **)&d_particle_soa.{{ p.name }}, {{ p.sizeof() }}));
  {% endfor %}
  {% for p in params if not p.is_type('P', 'RO') -%}
    ASSERT_NO_CUDA_ERROR(
      cudaMalloc((void **)&{{ p.device_name() }}, {{ p.sizeof() }}));
  {% endfor %}

  {% for p in params if p.is_type('-', 'RO') and not p.reload -%}
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_particle_soa.{{ p.name }}, {{ p.emit_name(name_prefix='h_') }}, {{ p.sizeof() }}, cudaMemcpyHostToDevice));
  {% endfor -%}
}

/*
 * (Re)fill neighbor list
 */
void {{name}}_update_neigh(
  int *h_numneigh,
  int **h_firstneigh,
  int **h_pages,
  {% for p in params if p.is_type('N', '-') -%} 
    {{ p.type }} **{{ p.emit_name(name_prefix='h_') }}{{ ',' if not loop.last }}
  {% endfor -%}
) {
  d_nl->reload(h_numneigh, h_firstneigh, h_pages);
  {% for p in params if p.is_type('N', '-') -%} 
    d_nl->load_pages<{{ p.type }}>({{ p.device_name() }}, {{ p.emit_name(name_prefix='h_') }}, {{ p.arity }});
  {% endfor -%}
}

void {{name}}_exit() {
  {% for p in params if p.is_type('P', 'RO') -%}
    assert(d_particle_soa.{{ p.name }});
  {% endfor %}
  assert(d_nl);
  {% for p in params if not p.is_type('P', 'RO') -%}
    assert({{ p.device_name() }});
  {% endfor %}

  {% for p in params if p.is_type('P', 'RO') -%}
    cudaFree(d_particle_soa.{{ p.name }});
  {% endfor %}
  delete(d_nl);
  {% for p in params if not p.is_type('P', 'RO') -%}
    cudaFree({{ p.device_name() }});
  {% endfor %}
}

void {{name}}_run(int N, int NSLOT, int maxpage, int pgsize,
  {% for p in params -%}
    {%- if p.is_type('-', 'RO') and p.reload -%}
      {{ p.emit_pointer_to_declaration(name_prefix='h_') }}{{ ',' if not loop.last }}
    {%- elif p.is_type('-', 'RW') or p.is_type('-', 'SUM') -%}
      {{ p.emit_pointer_to_declaration(name_prefix='h_') }}{{ ',' if not loop.last }}
    {%- else -%}
      //{{ p.name }}
    {%- endif %}
  {% endfor %}
  ) {

  {% for p in params if p.is_type('-', 'RO') and p.reload -%}
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy(d_particle_soa.{{ p.name }}, {{ p.emit_name(name_prefix='h_') }}, {{ p.sizeof() }}, cudaMemcpyHostToDevice));
  {% endfor %}
  {% for p in params if p.is_type('P', 'RW') or p.is_type('P', 'SUM') -%}
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy({{ p.device_name() }}, {{ p.emit_name(name_prefix='h_') }}, {{ p.sizeof() }}, cudaMemcpyHostToDevice));
  {% endfor %}

  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Pre-compute-kernel error: %s.\n", cudaGetErrorString(err));
    exit(1);
  }
#ifdef COMPUTE_TPA
  const int blockSize = 128;
  dim3 gridSize((N / blockSize)+1);
  {{name}}_tpa_compute_kernel<<<gridSize, blockSize>>>(
    N, d_particle_soa, 
    d_nl->d_numneigh, d_nl->d_pageidx, d_nl->d_offset, d_nl->pgsize, d_nl->d_neighidx,
    {% for p in params if not p.is_type('P', 'RO') -%}
      {{ p.device_name() }}{{ ',' if not loop.last }}
    {% endfor %}
  );
#else
  const int blockSize = NSLOT;
  dim3 gridSize(N);
  size_t sharedMemSize = 0;
  {% for p in params if p.is_type('P', 'SUM') -%}
    sharedMemSize += NSLOT * {{ p.arity }} * {{ p.sizeof_in_chars() }}; // {{ p.device_name() }}
  {% endfor %}
  {{name}}_bpa_compute_kernel<<<gridSize, blockSize, sharedMemSize>>>(
    N, NSLOT, d_particle_soa,
    d_nl->d_numneigh, d_nl->d_pageidx, d_nl->d_offset, d_nl->pgsize, d_nl->d_neighidx,
    {% for p in params if not p.is_type('P', 'RO') -%}
      {{ p.device_name() }}{{ ',' if not loop.last }}
    {% endfor %}
  );
#endif
  cudaThreadSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Post-compute-kernel error: %s.\n", cudaGetErrorString(err));
    exit(1);
  }

  {% for p in params if p.is_type('P', 'RW') or p.is_type('P', 'SUM') -%}
    ASSERT_NO_CUDA_ERROR(
      cudaMemcpy({{ p.emit_name(name_prefix='h_') }}, {{ p.device_name() }}, {{ p.sizeof() }}, cudaMemcpyDeviceToHost));
  {% endfor %}
}

#endif
