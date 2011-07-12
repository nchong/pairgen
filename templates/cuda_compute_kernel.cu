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
  struct neighbor neighbor_soa
  ) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    {% for p in params if p.is_type('P', 'RO') -%}
    {% if p.arity > 1 -%}
    {% for k in range(p.arity) -%}
    {{ "%s = neighbor_soa.%si[(idx*%d)+%d];" % (p.emit_assignment_i()[k], p.name, p.arity, k) }}
    {% endfor -%} {% else -%}
    {{ "%s = neighbor_soa.%si[idx];" % (p.emit_assignment_i()[0], p.name) }}
    {%- endif %}
    {% if p.arity > 1 -%}
    {% for k in range(p.arity) -%}
    {{ "%s = neighbor_soa.%sj[(idx*%d)+%d];" % (p.emit_assignment_j()[k], p.name, p.arity, k) }}
    {% endfor -%} {% else -%}
    {{ "%s = neighbor_soa.%sj[idx];" % (p.emit_assignment_j()[0], p.name) }}
    {%- endif %}
    {% endfor %}
    {% for p in params if p.is_type('P', 'RW') -%}
    {% if p.arity > 1 -%}
    {% for k in range(p.arity) -%}
    {{ "%s = neighbor_soa.%s_list[(idx*%d)+%d];" % (p.emit_assignment_i()[k], p.name, p.arity, k) }}
    {% endfor -%} {% else -%}
    {{ "%s = %s_list[idx];" % (p.emit_assignment_i()[0], p.name) }}
    {%- endif %}
    {% endfor %}
    {% for p in params if p.is_type('P', 'SUM') -%}
    {% if p.arity > 1 -%}
    {% for k in range(p.arity) -%}
    {{ "%s = neighbor_soa.%s_delta[(idx*%d)+%d];" % (p.emit_assignment_i()[k], p.name, p.arity, k) }}
    {% endfor -%} {% else -%}
    {{ "%s = %s_delta[idx];" % (p.emit_assignment_i()[0], p.name) }}
    {%- endif %}
    {% endfor %}
    {% for p in params if p.is_type('N', '-') -%}
    {% if p.arity > 1 -%}
    {% for k in range(p.arity) -%}
    {{ "%s = neighbor_soa.%s[(idx*%d)+%d];" % (p.emit_assignment_j()[k], p.name, p.arity, k) }}
    {% endfor -%} {% else -%}
    {{ "%s = %s[idx];" % (p.emit_assignment_j()[0], p.name) }}
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

    //writeback per-particle SUM data
    {% for p in params if p.is_type('P', 'SUM') -%}
    {% if p.arity > 1 -%}
    {% for k in range(p.arity) -%}
    {{ "neighbor_soa.%s[(idx*%d)+%d] = %s;" % (p.name, p.arity, k, p.emit_assignment_i()[k]) }}
    {% endfor -%} {% else -%}
    {{ "neighbor_soa.%s[idx] = %s;" % (p.name, p.emit_assignment_i()[0]) }}
    {%- endif %}
    {% endfor %}

    // writeback per-neighbor RW data
    {% for p in params if p.is_type('N', 'RW') -%}
    {% if p.arity > 1 -%}
    {% for k in range(p.arity) -%}
    {{ "neighbor_soa.%s[(neigh_idx*%d)+%d] = %s;" % (p.name, p.arity, k, p.emit_assignment_j()[k]) }}
    {% endfor -%} {% else -%}
    {{ "neighbor_soa.%s[neigh_idx] = %s;" % (p.name, p.emit_assignment_j()[0]) }}
    {%- endif %}
    {% endfor %}
  }
}

__global__ void {{name}}_cuda_collect_kernel(
  int P, // number of particles
  struct neighbor neighbor_soa,
  int *ioffset, int *icount, int *imapinv,
  int *joffset, int *jcount, int *jmapinv,
  //outputs
  {% for p in params if p.is_type('P', 'SUM') -%}
  {{ p.emit_pointer_to_declaration() }} {%- if not loop.last %}, {% endif %}
  {% endfor %}
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < P) {
    {% for p in params if p.is_type('P', 'SUM') -%}
    {{ "%s = %s;" % (p.emit_tagged_declaration()[0], p.emit_identity()) }}
    {% endfor %}
    int ioff = ioffset[idx];
    for (int i=0; i<icount[idx]; i++) {
      int e = imapinv[ioff+i];
      {% for p in params if p.is_type('P', 'SUM') -%}
      {% if p.arity > 1 -%}
      {% for k in range(p.arity) -%}
      {{ "%s[%d] += neighbor_soa.%s[(e*%d)+%d];" % (p.emit_tagged_name()[0], k, p.name, p.arity, k) }}
      {% endfor -%} {% else -%}
      {{ "%s += neighbor_soa.%s[e];" % (p.emit_tagged_name()[0], p.name) }}
      {%- endif %}
      {% endfor %}
    }

    int joff = joffset[idx];
    for (int i=0; i<jcount[idx]; i++) {
      int e = jmapinv[joff+i];
      {% for p in params if p.is_type('P', 'SUM') -%}
      {% if p.arity > 1 -%}
      {% for k in range(p.arity) -%}
      {{ "%s[%d] += neighbor_soa.%s[(e*%d)+%d];" % (p.emit_tagged_name()[0], k, p.name, p.arity, k) }}
      {% endfor -%} {% else -%}
      {{ "%s += neighbor_soa.%s[e];" % (p.emit_tagged_name()[0], p.name) }}
      {%- endif %}
      {% endfor %}
    }
  }
}

#endif
