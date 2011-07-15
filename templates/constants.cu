
void {{name}}_setup_constants(
  {% for c in consts -%}
    {{ c.type }} {{ c.emit_name(name_prefix='h_') }}{{ ',' if not loop.last }}
  {% endfor %}
  ) {
  {% for c in consts -%}
    cudaMemcpyToSymbol("{{ c.device_name() }}", &{{ c.emit_name(name_prefix='h_') }}, {{ c.sizeof() }}, 0, cudaMemcpyHostToDevice);
  {% endfor %}
}
