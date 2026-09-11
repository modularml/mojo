---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

## Documentation

## Language enhancements

## Language changes

## Library stabilizations

## Library changes

- `Coord.product()` has a new parameterized overload, `product[T: DType]()`,
  which accumulates and returns the product at `T` rather than at
  `Coord.DTYPE`:

  ```mojo
  var c = Coord(Idx[4], Int(8), Int32(3))
  var n = c.product[DType.uint32]()  # UInt32(96)
  ```

  The unparameterized `product()` is unchanged and still returns
  `Scalar[Coord.DTYPE]`. A `T` too narrow to hold the result wraps rather
  than widening, so a caller that picks one owns the overflow.

## GPU programming

## Tooling changes

## Removed

## Fixed
