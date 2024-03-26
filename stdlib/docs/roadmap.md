# Mojo Standard Library Roadmap

## Roadmap Cadence

The stdlib open-source repository has opted for a 6-month roadmap refresh
cadence that aligns with our internal workflows. The roadmap updates act as a
forcing function for discussions with the Mojo community to ensure stdlib
contributors both internally and externally are aligned on the future technical
direction.

## 2024 Q2+ Roadmap

The following are high-level themes the Mojo Standard Library team will be
working on over the next 6-months. Keep in mind that Mojo and the Mojo Standard
Library are in early development with many features landing over the months
ahead. Currently, that means we are focused on the core system programming
features that are essential to Mojo's mission.

### Core Library Improvements

- Remove `AnyRegType` in the Standard Library in favor of `AnyType`.

- Unify Pointer and AnyPointer.

- Apply `Reference` types and lifetimes throughout APIs and types.

- Design API conventions and expected behavior for core collection types such
  as `List`, `String`, `Dict` .

### Generic Programming Improvements

- Define core traits for generic programming models.

- Define an iterator model and implement iterator support for core types.

- Standardize collection meta type names (eg. *element_type*, *key_type*, and
  *value_type*).

### Improve Python Interop

- Improve `PythonObject` (including `object`) using new Mojo language
  features.

### Performance Improvements

- Set up performance benchmarking infrastructure along with regression tests for
  perf-sensitive data structures and algorithms.
