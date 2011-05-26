#include "{{ name }}_particle.h"
#include "{{ name }}_pair_kernel.cu"

/*
 * Cuda thread-per-particle decomposition.
 *
 * Given N particles,
 *  particle_soa.<var>[i]
 *  is the particle data for particle i
 *
 *  numneigh[i] 
 *  is the number of neighbors for particle i
 *
 *  neigh.<var>[(i*NSLOT)+jj] 
 *  is the particle data of the jj-th neighbor to particle i
 *
 * We assign one thread per particle i.
 * Each thread loops over the numneigh[i] neighbors of i.
 * NB: This kernel does not update any neighbor particles.
 *     Therefore the neighbor list must contain symmetric duplicates.
 */
__global__ void {{name}}_tpa_compute_kernel(
  int N, // number of particles
  int NSLOT, //number of data elements per particle
  struct particle particle_soa,
  int *numneigh,
  struct particle neigh
  {% for p in params if not p.is_type('P', 'RO') -%}
    , {{ p.emit_list_of_declaration() }} //list of length N*NSLOT*{{p.arity}}
  {% endfor %}
  ) {

  // register copies of particle and neighbor data
  {% for p in params if not p.is_type('P', 'SUM') -%}
    {% for v in p.emit_tagged_declaration() -%}
      {{ " " if not loop.first }}{{ v }};
    {%- endfor %}
  {% endfor -%}
  {% for p in params if p.is_type('P', 'SUM') -%}
    {{ "%s = %s;" % (p.emit_tagged_declaration()[0], p.emit_identity()) }}
  {% endfor %}
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N && numneigh[idx] > 0) {
    // load particle i data
    {% for p in params if p.is_type('P', 'RO') -%}
      {% if p.arity > 1 -%}
        {% for k in range(p.arity) -%}
          {{ "%s = particle_soa.%s[(idx*%d)+%d];" % (p.emit_assignment_i()[k], p.name, p.arity, k) }}
        {% endfor -%} {% else -%}
        {{ "%s = particle_soa.%s[idx];" % (p.emit_assignment_i()[0], p.name) }}
      {%- endif %}
    {% endfor %}
    {% for p in params if p.is_type('P', 'RW') -%}
      {% if p.arity > 1 -%}
        {% for k in range(p.arity) -%}
          {{ "%s = %s_list[(idx*%d)+%d];" % (p.emit_assignment_i()[k], p.name, p.arity, k) }}
        {% endfor -%} {% else -%}
          {{ "%s = %s_list[idx];" % (p.emit_assignment_i()[0], p.name) }}
      {%- endif %}
    {% endfor %}

    // iterate over each neighbor of particle i
    for (int jj=0; jj<numneigh[idx]; jj++) {
      // load particle j data
      int neigh_idx = (idx*NSLOT)+jj;
      // int j   = neigh.idx[neigh_idx];
      {% for p in params if p.is_type('P', 'RO') -%}
        {% if p.arity > 1 -%}
          {% for k in range(p.arity) -%}
            {{ "%s = neigh.%s[(neigh_idx*%d)+%d];" % (p.emit_assignment_j()[k], p.name, p.arity, k) }}
          {% endfor -%} {% else -%}
          {{ "%s = neigh.%s[neigh_idx];" % (p.emit_assignment_j()[0], p.name) }}
        {%- endif %}
      {% endfor %}
      {% for p in params if p.is_type('P', 'RW') -%}
        {% if p.arity > 1 -%}
          {% for k in range(p.arity) -%}
            {{ "%s = %s_list[(idx*%d)+%d];" % (p.emit_assignment_j()[k], p.name, p.arity, k) }}
          {% endfor -%} {% else -%}
            {{ "%s = %s_list[idx];" % (p.emit_assignment_j()[k], p.name) }}
        {%- endif %}
      {% endfor %}
      {% for p in params if p.is_type('N', 'RO') or p.is_type('N', 'RW') -%}
        {% if p.arity > 1 -%}
          {% for k in range(p.arity) -%}
            {{ "%s = %s_list[(neigh_idx*%d)+%d];" % (p.emit_assignment_j()[k], p.name, p.arity, k) }}
          {% endfor -%} {% else -%}
            {{ "%s = %s_list[neigh_idx];" % (p.emit_assignment_j()[0], p.name) }}
        {%- endif %}
      {% endfor %}

      // do pairwise calculation
      {{name}}_pair_kernel(
        {% for v in kernel_call_params -%}
          {{ v }} {%- if not loop.last %}, {% endif %}
        {% endfor -%}
      );

      // writeback per-particle and per-neighbor RW data
      {% for p in params if p.is_type('P', 'RW') -%}
        {% if p.arity > 1 -%}
          {% for k in range(p.arity) -%}
            {{ "%si_list[(idx*%d)+%d] = %s;" % (p.name, p.arity, k, p.emit_assignment_i()[k]) }}
          {% endfor -%} {% else -%}
            {{ "%si_list[idx] = %s;" % (p.name, p.arity, k, p.emit_assignment_i()[0]) }}
        {%- endif %}
      {% endfor %}
      {% for p in params if p.is_type('N', 'RW') -%}
        {% if p.arity > 1 -%}
          {% for k in range(p.arity) -%}
            {{ "%s_list[(neigh_idx*%d)+%d] = %s;" % (p.name, p.arity, k, p.emit_assignment_j()[k]) }}
          {% endfor -%} {% else -%}
            {{ "%s_list[neigh_idx] = %s;" % (p.name, p.emit_assignment_j()[0]) }}
        {%- endif %}
      {% endfor %}
    }

    //writeback per-particle SUM data
    {% for p in params if p.is_type('P', 'SUM') -%}
      {% if p.arity > 1 -%}
        {% for k in range(p.arity) -%}
          {{ "%si_list[(idx*%d)+%d] += %s;" % 
             (p.name, p.arity, k, p.emit_assignment_i()[k]) }}
        {% endfor -%} {% else -%}
          {{ "%si_list[idx] += %s;" % (p.name, p.emit_assignment_i()[0]) }}
      {%- endif %}
    {% endfor %}
  }
}
