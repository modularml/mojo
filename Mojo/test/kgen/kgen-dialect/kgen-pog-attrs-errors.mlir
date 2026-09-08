// RUN: kgen-opt %s -verify-parameters -verify-diagnostics -split-input-file -o /dev/null

// expected-error @+1 {{pos passing kind cannot follow pos_or_kw}}
#passing_kind_order1 = #kgen.pog_list<
  [<"a", pos_or_kw, not_vararg>, <"b", pos, not_vararg>, <"c", kw, not_vararg>, <"d", implicit, not_vararg>]
>

// -----

// expected-error @+1 {{pos_or_kw passing kind cannot follow implicit}}
#passing_kind_order2 = #kgen.pog_list<
  [<"a", pos, not_vararg>, <"b", kw, not_vararg>, <"c", implicit, not_vararg>, <"d", pos_or_kw, not_vararg>]
>

// -----

// expected-error @+1 {{kw passing kind cannot follow implicit}}
#passing_kind_order3 = #kgen.pog_list<
  [<"a", pos, not_vararg>, <"b", pos_or_kw, not_vararg>, <"c", implicit, not_vararg>, <"d", kw, not_vararg>]
>

// -----

// expected-error @+1 {{variadic convention not specified}}
#variadic_with_default = #kgen.pog_list<
  [<"a", pos, not_vararg>, <"b", pos_or_kw, pos_vararg, default :i8 1>, <"c", kw, not_vararg>, <"d", kw, not_vararg>]
>

// -----

// expected-error @+1 {{'inferred' parameter follows non-inferred parameter}}
#too_many_packs = #kgen.pog_list<
  [<"a", pos, not_vararg>, <"b", inferred, not_vararg>]
>

// -----

// expected-error @+1 {{default value of variadic pack must be UnknownAttr}}
#pack_with_default = #kgen.pog_list<
  [<"a", pos, not_vararg>, <"b", pos_or_kw, pack_vararg, default :i8 1>, <"c", kw, not_vararg>, <"d", kw, not_vararg>],
  owned_in_mem
>
