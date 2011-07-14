#ifndef {{ name|upper }}_CONSTANTS_H
#define {{ name|upper }}_CONSTANTS_H

{% for c in consts -%}
  __constant__ {{ "%s %s" % (c.type, c.device_name()) }};
{% endfor -%}

void {{name}}_setup_constants(
  {% for c in consts -%}
    {{ c.type }} {{ c.emit_name(name_prefix='h_') }}{{ ',' if not loop.last }}
  {% endfor %}
  );

#endif
