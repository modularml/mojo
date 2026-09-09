# Witness Elaboration

This doc goes over how Conformance Tables and Witness Entries are handled during
elaboration.

## Key Concepts

- StructGeneratorOp: A generic struct declaration containing the make up of the
type and all its conformance tables.

- ConformanceOp: A conformance table for a particular trait that this struct is
conforming to. It contains a witness entry for each requirement that the trait
has.

- WitnessOp: A witness entry. It provides a parameter value for a trait
requirement (identified by a name and a type). For example, for methods, it
might provide a GeneratorOp/FuncOp reference (as SymbolConstantAttr).

- StructInstanceOp: A concrete struct declaration containing the make up of the
type. At this moment it does not yet contain conformance tables since we have no
use for conformance tables in the run-time domain yet. Eagerly elaborating all
conformances will vastly increase the workload of elaboration just for them to
be discarded post-elaboration.

- TypeGeneratorRefAttr: A reference to a StructGeneratorOp. Can carry parameter
bindings for that generator. This will be evaluated to a TypeInstanceRefAttr
during elaboration.

- TypeInstanceRefAttr: A reference to a StructInstanceOp.

- GetWitnessAttr: A parameter operator that fetches a particular witness entry
from a struct type. It has three operands:
  - A type reference (parameter value).
  - A trait name (constant string).
  - A witness name (constant string).

## IREvaluator GetWitness Evaluation

Before elaboration, a GetWitnessAttr will likely contain references in the form
of a TypeGeneratorRefAttr.

During elaboration, these generic references will get automatically resolved to
concrete TypeInstanceRefAttrs. However, since they now reference the newly
created StructInstanceOps, which do not contain any witness tables yet, we
cannot directly perform witness lookups on these ops. Instead, we trace back to
the StructGeneratorOp that generated the instance to perform the lookup. This is
possible because the elaborator maintains this parent node relationship.

```text
                     GetWitnessAttr EVALUATION PROCESS

┌─────────────────────────────────────────────────────────────────────────────┐
│                            BEFORE ELABORATION                               │
│                                                                             │
│  GetWitnessAttr(                                                            │
│    type_ref: TypeGeneratorRefAttr ──────┐                                   │
│    trait_name: "Printable"              │                                   │
│    witness_name: "print"                │                                   │
│  )                                      │                                   │
│                                         │                                   │
│                                         ▼                                   │
│                            ┌─────────────────────────┐                      │
│                            │   StructGeneratorOp     │                      │
│                            │   "MyStruct<T>"         │                      │
│                            │                         │                      │
│                            │ ┌─────────────────────┐ │                      │
│                            │ │ ConformanceOp       │ │                      │
│                            │ │ trait: "Printable"  │ │                      │
│                            │ │                     │ │                      │
│                            │ │ ┌─────────────────┐ │ │                      │
│                            │ │ │ WitnessOp       │ │ │◄──── Direct lookup   │
│                            │ │ │ name: "print"   │ │ │                      │
│                            │ │ │ value: @func_x  │ │ │                      │
│                            │ │ └─────────────────┘ │ │                      │
│                            │ └─────────────────────┘ │                      │
│                            └─────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           DURING ELABORATION                                │
│                                                                             │
│  GetWitnessAttr(                                                            │
│    type_ref: TypeInstanceRefAttr ───────┐                                   │
│    trait_name: "Printable"              │                                   │
│    witness_name: "print"                │                                   │
│  )                                      │                                   │
│                                         │                                   │
│                                         ▼                                   │
│                            ┌─────────────────────────┐                      │
│                            │   StructInstanceOp      │                      │
│                            │   "MyStruct<Int>"       │                      │
│                            │                         │                      │
│                            │ ┌─────────────────────┐ │                      │
│                            │ │   Concrete Fields   │ │                      │
│                            │ │   (No conformance   │ │                      │
│                            │ │    tables!)         │ │                      │
│                            │ └─────────────────────┘ │                      │
│                            └─────────────────────────┘                      │
│                                         │                                   │
│                                         │ TRACE BACK via                    │
│                                         │ parent relationship               │
│                                         ▼                                   │
│                            ┌─────────────────────────┐                      │
│                            │   StructGeneratorOp     │                      │
│                            │   "MyStruct<T>"         │                      │
│                            │   (Original parent)     │                      │
│                            │                         │                      │
│                            │ ┌─────────────────────┐ │                      │
│                            │ │ ConformanceOp       │ │                      │
│                            │ │ trait: "Printable"  │ │                      │
│                            │ │                     │ │                      │
│                            │ │ ┌─────────────────┐ │ │                      │
│                            │ │ │ WitnessOp       │ │ │                      │
│                            │ │ │ name: "print"   │ │ │◄──── Lookup here!    │
│                            │ │ │ value: @func_x  │ │ │                      │
│                            │ │ └─────────────────┘ │ │                      │
│                            │ └─────────────────────┘ │                      │
│                            └─────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Slicing Out Offload Modules (WEASOOM)

In order for this to also work when we slice out pieces of the module for
compile-offload, we make sure to perform the same procedure there. We especially
need to watch out for TypeInstanceRefAttrs that are part of the parameter values
being used to instantiate an offloaded GeneratorOp.

Again, instead of slicing out the StructInstanceOp referenced by the
TypeInstanceRefAttr, we get the parent StructGeneratorOp and slice that out
instead. And following that, we replace the TypeInstanceRefAttrs in the
parameter values with their original TypeGeneratorRefAttrs. This way everything
also happens naturally in the offload module.
