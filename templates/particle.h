#ifndef {{ name|upper }}_PARTICLE_H
#define {{ name|upper }}_PARTICLE_H

struct particle {
  int *idx;
  {% for p in params if p.is_type('P', 'RO') -%}
    {{ p.emit_pointer_to_declaration() }}; //list of length N {{ "*%d" % p.arity if p.arity > 1 }}
  {% endfor %}
};

#endif
