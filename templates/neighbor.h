#ifndef {{ name|upper }}_NEIGHBOR_H
#define {{ name|upper }}_NEIGHBOR_H

/*
 * Struct of Array datastructure for neighbor data.
 */
struct neighbor {
#ifdef DEBUG
  int *i;
  int *j;
#endif
  //per-particle data
  {% for p in params if p.is_type('P', 'RO') -%}
  {{ p.emit_pointer_to_declaration(name_suffix="i") }}; //list of length N {{ "*%d" % p.arity if p.arity > 1 }}
  {{ p.emit_pointer_to_declaration(name_suffix="j") }}; //list of length N {{ "*%d" % p.arity if p.arity > 1 }}
  {% endfor %}
  {% for p in params if p.is_type('P', 'RW') -%}
  {{ p.emit_pointer_to_declaration() }}; //list of length N {{ "*%d" % p.arity if p.arity > 1 }}
  {% endfor %}
  {% for p in params if p.is_type('P', 'SUM') -%}
  {{ p.emit_pointer_to_declaration(name_suffix="_delta") }}; //list of length N {{ "*%d" % p.arity if p.arity > 1 }}
  {% endfor %}
  //per-neighbor data
  {% for p in params if p.is_type('N', '-') -%}
  {{ p.emit_pointer_to_declaration() }}; //list of length N {{ "*%d" % p.arity if p.arity > 1 }}
  {% endfor %}
};

#endif
