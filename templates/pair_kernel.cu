#ifndef {{ name|upper }}_PAIR_KERNEL_H
#define {{ name|upper }}_PAIR_KERNEL_H

#include "{{ name }}_constants.h"

#ifdef DEBUG
#include "cuPrintf.cu"
#define WATCHI 1
#define DEBUG_PRINT_INPUTS(printi) {   \
  if (i == printi || j == printi) {    \
    cuPrintf("i = %d; j= %d\n", i, j); \
    {% for p in params if p.is_type('P', 'RO')  or p.is_type('N', 'RO') -%}
      {% for v in p.emit_tagged_name() -%}
        {% if p.arity > 1 -%}
          {% for k in range(p.arity) -%}
            cuPrintf("{{ "%s[%d] = %s\\n" % (v, k, p.printf_format()) }}", {{ p.emit_assignment_i()[k] }}); \
          {% endfor %} {% else -%}
            cuPrintf("{{ "%s = %s\\n" % (v, p.printf_format()) }}", {{ p.emit_assignment_i()[0] }}); \
        {% endif -%}
      {% endfor -%}
    {% endfor -%}
    {% for p in params if p.is_type('P', 'RW')  or p.is_type('P', 'SUM') or p.is_type('N', 'RW') -%}
      {% for v in p.emit_tagged_name() -%}
        {% if p.arity > 1 -%}
          {% for k in range(p.arity) -%}
            cuPrintf("{{ "%s[%d] = %s\\n" % (v, k, p.printf_format()) }}", {{ "%s[%d]" % (p.emit_tagged_name()[0], k) }}); \
          {% endfor %} {% else -%}
            cuPrintf("{{ "%s = %s\\n" % (v, p.printf_format()) }}", {{ p.emit_tagged_name()[0] }}); \
        {% endif -%}
      {% endfor -%}
    {% endfor -%}
  }                                   \
} while(0);
#define DEBUG_PRINT_OUTPUTS(printi) { \
  if (i == printi || j == printi) {   \
    {% for p in params if p.is_type('P', 'RW')  or p.is_type('P', 'SUM') or p.is_type('N', 'RW') -%}
      {% for v in p.emit_tagged_name() -%}
        {% if p.arity > 1 -%}
          {% for k in range(p.arity) -%}
            cuPrintf("{{ "new %s[%d] = %s\\n" % (v, k, p.printf_format()) }}", {{ "%s[%d]" % (p.emit_tagged_name()[0], k) }}); \
          {% endfor %} {% else -%}
            cuPrintf("{{ "new %s = %s\\n" % (v, p.printf_format()) }}", {{ p.emit_tagged_name()[0] }}); \
        {% endif -%}
      {% endfor -%}
    {% endfor -%}
  }                                   \
} while(0);
#endif

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
#ifdef DEBUG
  int i, int j,
#endif
  {% for v in kernel_params -%}
    {{ v }} {%- if not loop.last %}, {% endif %}
  {% endfor -%}
) {
  //fill me in
}

#endif
