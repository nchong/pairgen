#ifndef {{ name|upper }}_BPA_COMPUTE_KERNEL_H
#define {{ name|upper }}_BPA_COMPUTE_KERNEL_H

#include "{{ name }}_particle.h"
#include "{{ name }}_pair_kernel.cu"

/*
 * Cuda unroll neighbor-list decomposition.
 *
 * Given N particles,
 *  particle_soa.<var#i>[v]
 *  particle_soa.<var#j>[v]
 *  is the particle data for particles i and j of neighbor v
 *
 * We assign one thread per particle neighbor n.
 * NB: This kernel does not update any neighbor particles.
 *     Therefore the neighbor list must contain symmetric duplicates.
 */

__global__ void {{name}}_cuda_compute_kernel(
  int N, // number of neighbors
  struct neighbor neighbor_soa,
  ) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    {% for p in params if p.is_type('P', 'RO') -%}
      {% if p.arity > 1 -%}
        {% for k in range(p.arity) -%}
          {{ "%s = neighbor_soa.%s[(idx*%d)+%d];" % (p.emit_assignment_i()[k], p.name, p.arity, k) }}
        {% endfor -%} {% else -%}
        {{ "%s = neighbor_soa.%s[idx];" % (p.emit_assignment_i()[0], p.name) }}
      {%- endif %}
    {% endfor %}
    {% for p in params if p.is_type('P', 'RW') -%}
      {% if p.arity > 1 -%}
        {% for k in range(p.arity) -%}
          {{ "%s = neighbor_soa.%s__list[(idx*%d)+%d];" % (p.emit_assignment_i()[k], p.name, p.arity, k) }}
        {% endfor -%} {% else -%}
          {{ "%s = %s_list[idx];" % (p.emit_assignment_i()[0], p.name) }}
      {%- endif %}
    {% endfor %}

    // do pairwise calculation
    {{name}}_pair_kernel(
#ifdef DEBUG
      neighbor_soa.i[idx], neighbor_soa.j[idx],
#endif
      {% for v in kernel_call_params -%}
        {{ v }} {%- if not loop.last %}, {% endif %}
      {% endfor -%}
    );
  }
}

#endif
