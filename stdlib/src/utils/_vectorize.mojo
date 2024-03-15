# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import align_down, is_power_of_2

# ===----------------------------------------------------------------------===#
# vectorize
# ===----------------------------------------------------------------------===#

# NOTE:
#   Per #34787, the source of truth for `vectorize` is here, but it is actually
#   documented and exposed from the `algorithms.functional` module.


@always_inline
fn vectorize[
    func: fn[width: Int] (Int) capturing -> None,
    simd_width: Int,
    /,
    *,
    unroll_factor: Int = 1,
](size: Int):
    var vector_end_simd = align_down(size, simd_width)
    _perfect_vectorized_impl[func, simd_width, unroll_factor=unroll_factor](
        size, vector_end_simd
    )

    for i in range(vector_end_simd, size):
        func[1](i)


@always_inline
fn vectorize[
    func: fn[width: Int] (Int) capturing -> None,
    simd_width: Int,
    /,
    *,
    size: Int,
    unroll_factor: Int = 1,
]():
    alias vector_end_simd = align_down(size, simd_width)
    _perfect_vectorized_impl[func, simd_width, unroll_factor](
        size, vector_end_simd
    )

    @parameter
    if size != vector_end_simd:

        @parameter
        if is_power_of_2(size - vector_end_simd):
            func[size - vector_end_simd](vector_end_simd)
        else:

            @unroll
            for i in range(vector_end_simd, size):
                func[1](i)


@always_inline
fn _perfect_vectorized_impl[
    func: fn[width: Int] (Int) capturing -> NoneType,
    /,
    *,
    simd_width: Int,
    unroll_factor: Int,
](size: Int, vector_end_simd: Int):
    constrained[simd_width > 0, "simd width must be > 0"]()
    constrained[unroll_factor > 0, "unroll factor must be > 0"]()

    alias unrolled_simd_width = simd_width * unroll_factor
    var vector_end_unrolled_simd = align_down(size, unrolled_simd_width)

    @always_inline
    @parameter
    fn unrolled_func(unrolled_simd_idx: Int):
        @unroll
        for idx in range(unroll_factor):
            func[simd_width](unrolled_simd_idx + idx * simd_width)

    for unrolled_simd_idx in range(
        0, vector_end_unrolled_simd, unrolled_simd_width
    ):
        unrolled_func(unrolled_simd_idx)

    @parameter
    if unroll_factor != 1:
        for simd_idx in range(
            vector_end_unrolled_simd, vector_end_simd, simd_width
        ):
            func[simd_width](simd_idx)
