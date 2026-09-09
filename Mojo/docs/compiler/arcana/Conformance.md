# Conformance

This doc goes over how struct-trait conformances are checked and materialized
in the parser.

## Key Concepts

- StructDeclOp: A struct declaration containing aliases, fields, methods, etc.
and conformances.

- TraitDeclOp: A trait declaration containing associated aliases & methods.

- ConformanceOp: A table that tracks how a struct conforms to a trait. For
every struct, and for every trait that the struct conforms to, there is a
ConformanceOp. The name of the ConformanceOp is the mangled name of the trait.

- WitnessOp: A specific entry in a ConformanceOp that maps a name & type pair
(required by the trait) to an implementation (provided by the struct). For
example, if the trait requires a particular function (with a particular name &
type), there will be a WitnessOp in the ConformanceOp that has that name, and
has a symbol reference to the actual function declared in the struct that
satisfies that requirement.

## Lazy Resolution of Conformances (CALROC)

Like most other things in the parser, conformances are also lazily resolved.

ConformanceOps are modeled as ASTDecls inside StructDeclOps:

- When a StructDeclOp is fully resolved, a nested ConformanceOp is created for
each explicit conformance that the struct declares. At this point, they have
empty bodies (marked as signature-resolved).

- When checking conformance (doesNominalTypeConformTo), we ask to body resolve
the conformance table for that trait, and all its parent traits, in that
struct. This means only witness tables that are needed by the current program
is body resolved.

- At the end of parsing, all reachable decls that are signature resolved will
be automatically fully resolved. This will ensure every explicit conformance
declared by a struct in the "main" module is checked.

Note that ConformanceOps are always created as signature-resolved because there
is no less-resolved state for it (it is a synthesized entity after all).

When creating packages, all conformance tables for every trait a struct
conforms to are created in it (as usual). When loading a StructDeclOp from
bytecode, we follow a similar process:

- When a bytecode StructDeclOp is fully materialized, its inner ConformanceOps
are created as signature resolved, and have empty bodies (non-materialized
yet).

- When a bytecode ConformanceOp is fully materialized, its inner WitnessOps are
materialized and any references are recursively resolved as usual. This loads
in any witnessing functions for example.
