# Mojo standard library roadmap

This is a high-level plan for the Mojo standard library.

We plan to update this roadmap approximately every 6 months, in alignment with
Modular's internal workflows. The roadmap updates act as a forcing function for
discussions with the Mojo community to ensure the standard library contributors
both internally and externally are aligned on the future technical direction.

For more about our long-term aspirations, check out our [Vision doc](vision.md).

## 2024 Q2+ roadmap

The following are high-level themes the Mojo standard library team will be
working on over the next 6 months. Keep in mind that Mojo and the Mojo standard
library are still in early development and many features will land in the
months ahead. Currently, that means we are focused on the core system
programming features that are essential to [Mojo's
mission](https://docs.modular.com/mojo/why-mojo).

### Core library improvements

- Remove `AnyTrivialRegType` in the standard library in favor of `AnyType`.

- Unify `Pointer` and `AnyPointer`.

- Apply `Reference` types and lifetimes throughout APIs and types.

- Design API conventions and expected behavior for core collection types such
  as `List`, `String`, and `Dict`.

### Generic programming improvements

- Define core traits for generic programming models.

- Define an iterator model and implement iterator support for core types.

- Standardize collection meta type names (such as *element_type*, *key_type*,
  and *value_type*).

### Improve Python interop

- Improve `PythonObject` (including `object`) using new Mojo language
  features.

### Performance improvements

- Set up performance benchmarking infrastructure along with regression tests for
  perf-sensitive data structures and algorithms.
