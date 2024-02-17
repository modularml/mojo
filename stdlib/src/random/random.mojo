# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides functions for random numbers.

You can import these APIs from the `random` package. For example:

```mojo
from random import seed
```
"""

from sys import external_call
from sys.info import bitwidthof
from time import now

from memory.unsafe import DTypePointer
from tensor import Tensor, TensorShape, TensorSpec


fn _get_random_state() -> DTypePointer[DType.invalid]:
    return external_call[
        "KGEN_CompilerRT_GetRandomState",
        DTypePointer[DType.invalid],
    ]()


fn seed():
    """Seeds the random number generator using the current time."""
    seed(now())


fn seed(a: Int):
    """Seeds the random number generator using the value provided.

    Args:
        a: The seed value.
    """
    external_call["KGEN_CompilerRT_SetRandomStateSeed", NoneType](
        _get_random_state(), a
    )


fn random_float64(min: Float64 = 0, max: Float64 = 1) -> Float64:
    """Returns a random `Float64` number from the given range.

    Args:
        min: The minimum number in the range (default is 0.0).
        max: The maximum number in the range (default is 1.0).

    Returns:
        A random number from the specified range.
    """
    return external_call["KGEN_CompilerRT_RandomDouble", Float64](
        _get_random_state(), min, max
    )


fn random_si64(min: Int64, max: Int64) -> Int64:
    """Returns a random `Int64` number from the given range.

    Args:
        min: The minimum number in the range.
        max: The maximum number in the range.

    Returns:
        A random number from the specified range.
    """
    return external_call["KGEN_CompilerRT_RandomSInt64", Int64](
        _get_random_state(), min, max
    )


fn random_ui64(min: UInt64, max: UInt64) -> UInt64:
    """Returns a random `UInt64` number from the given range.

    Args:
        min: The minimum number in the range.
        max: The maximum number in the range.

    Returns:
        A random number from the specified range.
    """
    return external_call["KGEN_CompilerRT_RandomUInt64", UInt64](
        _get_random_state(), min, max
    )


fn randint[
    type: DType
](ptr: DTypePointer[type], size: Int, low: Int, high: Int):
    """Fills memory with uniform random in range [low, high].

    Constraints:
        The type should be integral.

    Parameters:
        type: The dtype of the pointer.

    Args:
        ptr: The pointer to the memory area to fill.
        size: The number of elements to fill.
        low: The minimal value for random.
        high: The maximal value for random.
    """
    constrained[type.is_integral(), "type must be integral"]()

    @parameter
    if type.is_signed():
        for si in range(size):
            ptr[si] = random_si64(low, high).cast[type]()
    else:
        for ui in range(size):
            ptr[ui] = random_ui64(low, high).cast[type]()


fn rand[type: DType](ptr: DTypePointer[type], size: Int):
    """Fills memory with random values from a uniform distribution.

    Parameters:
        type: The dtype of the pointer.

    Args:
        ptr: The pointer to the memory area to fill.
        size: The number of elements to fill.
    """
    alias bitwidth = bitwidthof[type]()

    @parameter
    if type.is_floating_point():
        for i in range(size):
            ptr[i] = random_float64().cast[type]()
        return

    @parameter
    if type == DType.bool:
        for i in range(size):
            ptr[i] = random_ui64(0, 1).cast[type]()
        return

    @parameter
    if type.is_signed():
        for i in range(size):
            ptr[i] = random_si64(
                -(1 << (bitwidth - 1)), (1 << (bitwidth - 1)) - 1
            ).cast[type]()
        return

    @parameter
    if type.is_unsigned():
        for i in range(size):
            ptr[i] = random_ui64(0, (1 << bitwidth) - 1).cast[type]()
        return


fn rand[type: DType](*shape: Int) -> Tensor[type]:
    """Constructs a new tensor with the specified shape and fills it with random
    elements.

    Parameters:
        type: The dtype of the tensor.

    Args:
        shape: The tensor shape.

    Returns:
        A new tensor of specified shape and filled with random elements.
    """
    return rand[type](TensorShape(shape))


fn rand[type: DType](owned shape: TensorShape) -> Tensor[type]:
    """Constructs a new tensor with the specified shape and fills it with random
    elements.

    Parameters:
        type: The dtype of the tensor.

    Args:
        shape: The tensor shape.

    Returns:
        A new tensor of specified shape and filled with random elements.
    """
    let tensor = Tensor[type](shape ^)
    rand(tensor.data(), tensor.num_elements())
    return tensor


fn rand[type: DType](owned spec: TensorSpec) -> Tensor[type]:
    """Constructs a new tensor with the specified specification and fills it
    with random elements.

    Parameters:
        type: The dtype of the tensor.

    Args:
        spec: The tensor specification.

    Returns:
        A new tensor of specified specification and filled with random elements.
    """
    let tensor = Tensor[type](spec ^)
    rand(tensor.data(), tensor.num_elements())
    return tensor


fn randn_float64(mean: Float64 = 0.0, variance: Float64 = 1.0) -> Float64:
    """Returns a random double sampled from Normal(mean, variance) distribution.

    Args:
        mean: Normal distribution mean.
        variance: Normal distribution variance.

    Returns:
        A random float64 sampled from Normal(mean, variance).
    """
    return external_call["KGEN_CompilerRT_NormalDouble", Float64](
        _get_random_state(), mean, variance
    )


fn randn[
    type: DType
](
    ptr: DTypePointer[type],
    size: Int,
    mean: Float64 = 0.0,
    variance: Float64 = 1.0,
):
    """Fills memory with random values from a Normal(mean, variance) distribution.

    Constraints:
        The type should be floating point.

    Parameters:
        type: The dtype of the pointer.

    Args:
        ptr: The pointer to the memory area to fill.
        size: The number of elements to fill.
        mean: Normal distribution mean.
        variance: Normal distribution variance.
    """

    for i in range(size):
        ptr[i] = randn_float64(mean, variance).cast[type]()
    return


fn randn[
    type: DType
](
    owned shape: TensorShape,
    mean: Float64 = 0.0,
    variance: Float64 = 1.0,
) -> Tensor[type]:
    """Constructs a new Tensor from the shape and fills it with random values from a Normal(mean, variance) distribution.

    Constraints:
        The type should be floating point.

    Parameters:
        type: The dtype of the pointer.

    Args:
        shape: The shape of the Tensor to fill with random values.
        mean: Normal distribution mean.
        variance: Normal distribution variance.

    Returns:
        A Tensor filled with random dtype samples from Normal(mean, variance).
    """

    let tensor = Tensor[type](shape ^)
    randn(tensor.data(), tensor.num_elements(), mean, variance)
    return tensor


fn randn[
    type: DType
](
    owned spec: TensorSpec,
    mean: Float64 = 0.0,
    variance: Float64 = 1.0,
) -> Tensor[type]:
    """Constructs a new Tensor from the spec and fills it with random values from a Normal(mean, variance) distribution.

    Constraints:
        The type should be floating point.

    Parameters:
        type: The dtype of the pointer.

    Args:
        spec: The spec of the Tensor to fill with random values.
        mean: Normal distribution mean.
        variance: Normal distribution variance.

    Returns:
        A Tensor filled with random dtype samples from Normal(mean, variance).
    """

    let tensor = Tensor[type](spec ^)
    randn(tensor.data(), tensor.num_elements(), mean, variance)
    return tensor
