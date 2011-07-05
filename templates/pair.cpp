#include "pair_{{ name }}.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"

// external functions from pairgen library
void {{name}}_init(int N, int NSLOT,
  {% for p in params if p.is_type('-', 'RO') and not p.reload -%}
    {{ p.emit_pointer_to_declaration(name_prefix='h_') }}{{ ',' if not loop.last }}
  {% endfor -%}
);
void {{name}}_exit();
void {{name}}_run(int N, int NSLOT,
  int *h_numneigh,
  int *h_neighidx,
  {% for p in params -%}
    {%- if p.is_type('-', 'RO') and p.reload -%}
      {{ p.emit_pointer_to_declaration(name_prefix='h_') }}{{ ',' if not loop.last }}
    {%- elif p.is_type('-', 'RW') or p.is_type('-', 'SUM') -%}
      {{ p.emit_pointer_to_declaration(name_prefix='h_') }}{{ ',' if not loop.last }}
    {%- else -%}
      //{{ p.name }}
    {%- endif %}
  {% endfor %}
);

using namespace LAMMPS_NS;

Pair{{ name|capitalize }}::Pair{{ name|capitalize }}(LAMMPS *lmp) : Pair(lmp) {
  if (!(force->pair_match("pairgen/{{ name }}", 0))) return;
}

Pair{{ name|capitalize}}::~Pair{{ name|capitalize }}() {

}

void Pair{{ name|capitalize }}::init_style() {
  // request a full neighbor list
  int irequest = neighbor->request(this);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;

  // request a neighbor list for each per-neighbor parameter
  {% for p in params if (p.is_type('N', 'RO') or p.is_type('N', 'RW')) and p.name != 'touch' -%}
    // request a half neighbor list for {{ p.name }}
    irequest = neighbor->request(this);
    neighbor->requests[irequest]->id   = {{ loop.index }};
    neighbor->requests[irequest]->half = 1;
    neighbor->requests[irequest]->full = 0;
    neighbor->requests[irequest]->dnum = {{ p.arity }};
  {% endfor %}
}

void Pair{{ name|capitalize }}::init_list(int id, NeighList *ptr)
{
  if (id == 0) { //neighbor list
    list = ptr;
  }
  {% for p in params if (p.is_type('N', 'RO') or p.is_type('N', 'RW')) and p.name != 'touch' -%}
  if (id == {{ loop.index }}) {
    {{ p.name }}_list = ptr;
  }
  {% endfor %}
}

void Pair{{ name|capitalize }}::compute(int eflag, int vflag) {

}

void Pair{{ name|capitalize }}::settings(int narg, char **arg) {

}

void Pair{{ name|capitalize }}::coeff(int narg, char **arg) {

}
