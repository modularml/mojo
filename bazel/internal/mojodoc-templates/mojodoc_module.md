<!-- rumdl-disable -->
{% import 'macros.jinja' as macros %}
{# Print YAML front matter #}
{% macro print_front_matter(decl) %}
---
title: {{ decl.name }}
{% if decl.sidebar_label %}sidebar_label: {{ decl.sidebar_label }}
{% endif %}
version: {{ decl.version }}
{% if decl.packages or decl.modules %}type: package
{% else %}type: module
{% if decl.module_name %}module_name: {{ decl.module_name }}
{% endif %}
namespace: {{ decl.namespace }}
{% endif %}
{% if decl.packages and not decl.modules %}sidebar_position: 1
{% endif %}
lang: mojo
description: {% if decl.summary
  %}"{{ macros.escape_quotes(decl.summary) }}"
  {% else %}"Mojo {%
    if decl.packages or decl.modules %}package{% else %}module{%
    endif %} {{ decl.namespace }}.{{ decl.name }} documentation"
  {% endif %}
---

<section class='mojo-docs'>

{% endmacro -%}
{# Print each declaration #}
{% macro process_decl_body(decl) %}

{{ decl.summary }}

{{ decl.description }}

{% endmacro %}
{#############}
{# Main loop #}
{#############}
{% for decl in decls recursive %}
{% if loop.depth == 1 %}
{{ print_front_matter(decl) }}
{% endif %}

<div class='mojo-module-detail'><!-- here only for Listing component -->

{{ process_decl_body(decl) }}

</div>

{% if decl.aliases %}

## `comptime` values

{% for alias in decl.aliases | sort(attribute='name') -%}

{# Extra div to flex align stability marker. #}
<div class='mojo-alias-header'>

###  `{{ alias.name }}`

{{ macros.stability_marker(alias, header=True) }}

</div>

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">

{# For values that could contain IR (signatures, types, values), use #}
{# double backticks to preserve literal backticks. #}
{# Spaces between the double-backticks and content need to be balanced, #}
{# so we either add them manually (as here) or use pad_backticks filter. #}

`` {{ alias.signature }} = {{ alias.value }} ``

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
        {# Trait names should never contain backticks, so no double backticks here. #}
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
{% if decl.packages %}

## Packages

{% for package in decl.packages | sort(attribute='name') -%}
* [​`{{ package.name }}`](./{{ package.name }}/): {{ package.summary }}
{% endfor %}
{% endif %}
{% if decl.modules %}

## Modules

{% for module in decl.modules | sort(attribute='name') -%}
* [​`{{ module.name }}`](./{{ module.slug }}/): {{ module.summary }}
{% endfor %}
{% endif %}
{% if decl.structs %}

## Structs

{% for struct in decl.structs | sort(attribute='name') -%}
* [​`{{ struct.name }}`](./{{ struct.name }}): {{ struct.summary }}
{% endfor %}
{% endif %}
{% if decl.traits %}

## Traits

{% for trait in decl.traits | sort(attribute='name') -%}
* [​`{{ trait.name }}`](./{{ trait.name }}): {{ trait.summary }}
{% endfor %}
{% endif %}
{% if decl.functions %}

## Functions

{% for function in decl.functions | sort(attribute='name') -%}
* [​`{{ function.name }}`](./{{ function.filename }}): {{ function.overloads[0].summary }}
{% endfor %}
{% endif %}
{% endfor %}

</section>
