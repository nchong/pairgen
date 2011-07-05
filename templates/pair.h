#ifdef PAIR_CLASS

PairStyle(pairgen/{{ name }},Pair{{ name|capitalize }})

#else

#ifndef LMP_PAIR_{{ name|upper }}_H
#define LMP_PAIR_{{ name|upper }}_H
#include "pair.h"

namespace LAMMPS_NS {

class Pair{{ name|capitalize }} : public Pair {
 public:
    Pair{{ name|capitalize }}(class LAMMPS *);
    ~Pair{{ name|capitalize }}();
    void compute(int, int);
    void settings(int, char **);
    void coeff(int, char **);
    void init_style();
    void init_list(int id, NeighList *ptr);

  {% for p in params if (p.is_type('N', 'RO') or p.is_type('N', 'RW')) and p.name != 'touch' -%}
    NeighList *{{ p.name }}_list;
  {% endfor %}

};

}

#endif

#endif
