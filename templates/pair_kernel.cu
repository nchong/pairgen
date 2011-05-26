#ifndef {{ name|upper }}_PAIR_KERNEL_H
#define {{ name|upper }}_PAIR_KERNEL_H

#include "{{ name }}_constants.h"

/*
 * Pairwise interaction of particles i and j
 * Constants:
 {% for c in consts -%}
 * {{ "\t%s" % c.device_name() }} {{ "\t-- %s" % c.description if c.description }}
 {% endfor -%}
 * Read-Only:
 {% for p in params if p.is_type('P', 'RO')  or p.is_type('N', 'RO') -%}
 * {{"\t"}} {%-   for v in p.emit_tagged_name() -%}
   {{ "%s" % v }} {%- if not loop.last %}, {% endif %}
   {%- endfor -%}
   {{- "\t-- %s" % p.description if p.description }}
 {% endfor -%}
 * Update:
 {% for p in params if p.is_type('P', 'RW')  or p.is_type('P', 'SUM') or p.is_type('N', 'RW') -%}
 * {{"\t"}} {%-   for v in p.emit_tagged_name() -%}
   {{ "%s" % v }} {%- if not loop.last %}, {% endif %}
   {%- endfor -%}
   {{- "\t-- %s" % p.description if p.description }}
 {% endfor -%}
 */
__device__ void {{name}}_pair_kernel(
  {% for v in kernel_params -%}
    {{ v }} {%- if not loop.last %}, {% endif %}
  {% endfor -%}
) {
  //fill me in
}

#endif
