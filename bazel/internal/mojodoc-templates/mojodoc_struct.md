<!-- rumdl-disable -->
{# Print YAML front matter #}
{% import 'macros.jinja' as macros %}
{% macro print_front_matter(decl) %}
---
title: {{ decl.name }}
{% if decl.sidebar_label %}sidebar_label: {{ decl.sidebar_label }}
{% endif %}
version: {{ decl.version }}
slug: {{ decl.slug }}
type: {{ decl.kind }}
{% if decl.module_name %}module_name: {{ decl.module_name }}
{% endif %}
namespace: {{ decl.namespace }}
lang: mojo
show_stability_marker: {{ decl.showStabilityMarker }}
{% if decl.isStable %}is_stable: true
{% endif %}
{% if decl.sinceVersion %}since_version: {{ decl.sinceVersion }}
{% endif %}
description: {% if decl.summary
  %}"{{ macros.escape_quotes(decl.summary) }}"
  {% else %}"Mojo {{ decl.kind }} `{{ decl.namespace }}.{{ decl.name }}` documentation"
  {% endif %}
---

<section class='mojo-docs'>

{% endmacro -%}
{# Print each declaration #}
{% macro process_decl_body(decl, marker_above=False) %}
{# The page's own declaration wears its marker above the signature, clear of #}
{# the code font; its members carry theirs inside the signature row. #}
{% if marker_above %}
{{ macros.stability_marker(decl, standalone=True) }}
{% endif %}

{% if decl.signature %}
<div class="mojo-function-sig">

{% if decl.implicit %}
`@implicit`
{% endif %}
{# For values that could contain IR (signatures, types, values), use #}
{# double backticks to preserve literal backticks. #}
{# Spaces between the double-backticks and content need to be balanced, #}
{# so we either add them manually (as here) or use pad_backticks filter. #}
{% if decl.signature %}
`` {% if decl.isStatic %}static {% endif %}{{ decl.signature }} ``
{% endif %}
{% if not marker_above %}
{{ macros.stability_marker(decl) }}
{% endif %}

</div>
{% endif %}

{{ decl.summary }}

{{ decl.description }}

{% if decl.deprecated %}

**Deprecated:** {{ decl.deprecated }}
{% endif %}

{% if decl.constraints %}

**Constraints:**

{{ decl.constraints }}
{% endif %}
{% if decl.parameters and decl.kind == 'function' %}

**Parameters:**

{% for param in decl.parameters -%}
*   ​<b>{{ param.name }}</b> ({% if param.traits -%}
        {# Trait names should never contain backticks, so no double backticks here. #}
        {%- for trait in param.traits
            %}{{ api_link(trait.type, trait.path) }}
            {%- if not loop.last %} & {% endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ api_link(param.type, param.path, padding=True) }}
    {%- endif %}): {{ param.description }}
{% endfor %}
{% endif %}
{% if decl.args %}

**Args:**

{% for arg in decl.args -%}
*   ​<b>{{ arg.name }}</b> ({{ api_link(arg.type, arg.path, padding=True) }}): {{ arg.description }}
{% endfor %}
{% endif %}
{% if (decl.returns and decl.returns.type != 'Self') or (decl.returns and decl.returns.doc) %}
{# Don't show "Returns" if the type is Self, unless there's a docstring #}

**Returns:**

{{ api_link(decl.returns.type, decl.returns.path, padding=True) }}{% if decl.returns.doc
    %}: {{ decl.returns.doc }}{% endif %}
{% endif %}
{% if decl.raisesDoc %}

**Raises:**

{{ decl.raisesDoc }}
{% endif %}
{% endmacro %}
{#############}
{# Main loop #}
{#############}
{% for decl in decls recursive %}
{% set outer_loop = loop %}
{% if loop.depth == 1 %}
{{ print_front_matter(decl) }}
{% else %}
{{ "#"*(loop.depth+1) }} `{{ decl.name }}`

{% endif %}
{% if decl.overloads %}
{% for overload in decl.overloads %}
<div class='mojo-function-detail'>

{{ process_decl_body(overload) }}

</div>

{% endfor %}
{% else %}

{{ process_decl_body(decl, marker_above=True) }}

{% endif %}

{% if decl.parameters and not decl.kind == 'function' %}

## Parameters

{% for param in decl.parameters -%}
*   ​<b>{{ param.name }}</b> ({% if param.traits -%}
        {%- for trait in param.traits
            %}{{ api_link(trait.type, trait.path) }}
            {%- if not loop.last %} & {% endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ api_link(param.type, param.path, padding=True) }}
    {%- endif %}): {{ param.description }}
{% endfor %}
{% endif %}
{% if decl.fields %}

## Fields

{% for field in decl.fields %}
* ​<b>{{ field.name }}</b> (``{{field.type | pad_backticks }}``): {{ field.summary }}
{% if field.description %}
{{field.description | indent(2, True, False)}}
{% endif %}
{% endfor %}
{% endif %}
{% if decl.parentTraits %}
{% if decl.kind == 'struct' or decl.kind == 'trait' %}

## Implemented traits

{% for trait in decl.parentTraits %}
{{ api_link(trait.name, trait.path) }}{% if trait.condition %} (`where {{ trait.condition }}`){% endif %}{{ ", " if not loop.last else "" }}
{% endfor %}

{% endif %}
{% endif %}
{% if decl.aliases %}

## `comptime` members

{% for alias in decl.aliases | sort(attribute='name') %}

{# Extra div to flex align stability marker. #}
<div class='mojo-alias-header'>

###  `{{ alias.name }}`

{{ macros.stability_marker(alias, header=True) }}

</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">

{# don't show value for trait aliases (no value) or if name == value #}
{% if alias.value and alias.name != alias.value %}
`` {{ alias.signature }} = {{ alias.value }} ``
{% else %}
`{{ alias.signature }}`
{% endif %}

</div>

{% if alias.summary %}
{{ alias.summary }}

{% endif %}
{% if alias.description %}
{{alias.description | indent(2, True, False)}}
{% endif %}
{% if alias.deprecated %}
**Deprecated:** {{ alias.deprecated }}
{% endif %}

{% if alias.parameters %}

#### Parameters

{% for param in alias.parameters -%}
*   ​<b>{{ param.name }}</b> ({% if param.traits -%}
        {%- for trait in param.traits
            %}{{ api_link(trait.type, trait.path) }}{%
            if not loop.last %} & {% endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ api_link(param.type, param.path, padding=True) }}
    {%- endif %}): {{ param.description }}
{% endfor %}
{% endif %}
</div>

{% endfor %}
{% endif %}
{% if decl.kind == 'trait' %}
{% if decl.required_methods %}

## Required methods

{{ outer_loop(decl.required_methods) }}
{% endif %}
{% if decl.provided_methods %}

## Provided methods

{{ outer_loop(decl.provided_methods) }}
{% endif %}
{% else %}
{% if decl.functions %}

## Methods

{{ outer_loop(decl.functions) }}
{% endif %}
{% endif %}
{% endfor %}

</section>
