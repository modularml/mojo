# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements functions for retrieving compile-time defines.

You can use these functions to set parameter values or runtime constants based on
name-value pairs defined on the command line. For example:

  ```mojo
  from sys.param_env import is_defined
  from tensor import Tensor, TensorSpec

  alias float_type: DType = DType.float32 if is_defined["FLOAT32"]() else DType.float64
  
  let spec = TensorSpec(float_type, 256, 256)
  var image = Tensor[float_type](spec)
```

And on the command line:

```
  mojo -D FLOAT_32 main.mojo
```

For more information, see the [Mojo build docs](https://docs.modular.com/mojo/cli/build.html#d-keyvalue).
The `mojo run` command also supports the `-D` option.


You can import these APIs from the `sys` package. For example:

```mojo
from sys.param_env import is_defined
```
"""


fn is_defined[name: StringLiteral]() -> Bool:
    """Return true if the named value is defined.

    Parameters:
        name: The name to test.

    Returns:
        True if the name is defined.
    """
    alias result = __mlir_attr[
        `#kgen.param.expr<get_env, `, name.value, `> : i1`
    ]
    return result


fn env_get_int[name: StringLiteral]() -> Int:
    """Try to get an integer-valued define. Compilation fails if the
    name is not defined.

    Parameters:
        name: The name of the define.

    Returns:
        An integer parameter value.
    """
    alias result = __mlir_attr[
        `#kgen.param.expr<get_env, `, name.value, `> : index`
    ]
    return result


fn env_get_int[name: StringLiteral, default: Int]() -> Int:
    """Try to get an integer-valued define. If the name is not defined, return
    a default value instead.

    Parameters:
        name: The name of the define.
        default: The default value to use.

    Returns:
        An integer parameter value.
    """

    @parameter
    if is_defined[name]():
        return env_get_int[name]()
    else:
        return default


fn env_get_string[name: StringLiteral]() -> StringLiteral:
    """Try to get a string-valued define. Compilation fails if the
    name is not defined.

    Parameters:
        name: The name of the define.

    Returns:
        A string parameter value.
    """
    alias result = __mlir_attr[
        `#kgen.param.expr<get_env, `, name.value, `> : !kgen.string`
    ]
    return result


fn env_get_string[
    name: StringLiteral, default: StringLiteral
]() -> StringLiteral:
    """Try to get a string-valued define. If the name is not defined, return
    a default value instead.

    Parameters:
        name: The name of the define.
        default: The default value to use.

    Returns:
        A string parameter value.
    """

    @parameter
    if is_defined[name]():
        return env_get_string[name]()
    else:
        return default
