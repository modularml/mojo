# Mojo docstring style guide

This is a language style guide for Mojo API docs (code comments known as
“docstrings”). The Mojo docstring style is based on the
[Google Python docstring style](https://google.github.io/styleguide/pyguide.html#381-docstrings),
with the addition of the Mojo-specific section headings, `Parameters:` and
`Constraints:`.

This is a brief set of guidelines that cover most situations. If you have
questions that are not answered here, refer to the more comprehensive [Google
Style Guide for API reference code
comments](https://developers.google.com/style/api-reference-comments).

For information on validating docstrings, see
[API docstrings](style-guide.md#api-docstrings) in the Coding standards and
style guide.

## Basics

- Docstrings support Markdown formatting.

- End all sentences with a period (including sentence fragments).

  - As you’ll see, most API descriptions are sentence fragments (they are
    often missing a subject because we don’t repeat the struct, function, or
    argument name in the first sentence).

- Use code font for all API names (structs, functions, attributes, argument and
  parameter names, etc.).

  - Create code font with backticks (\`Int\`).

  - Include empty parentheses for function/method names, regardless of
    argument length, but don't include square brackets (even if the function
    takes parameters) If it's crucial to identify a specific function overload,
    add argument names, and/or a parameter list.

    For example:

    - Call the `erase()` method.

    - Use `pop(index)` to pop a specific element from the list.

    - If you know the power at compile time, you can use the `pow[n](x)` version
      of this function.

## Functions/Methods

### Description

- The first sentence is a brief description of what the *function* *does*. The
  first word should be a present tense verb ("Gets", "Sets", "Checks",
  "Converts", "Performs", "Adds", etc.).

- If you’re unsure how to phrase a description, just answer the the question,
  “What does this function do?” Your answer should complete the sentence, “This
  function ____” (but without saying “this function”).

  - A blank line follows the first sentence.

- If there are any prerequisites, specify them with the second
  sentence. Then provide a more detailed description, if necessary.  )

### Parameters, arguments, and return values

- Use a noun phrase to describe what the *argument or parameter is.* This
  description should be formatted as a sentence (capitalize the first word, add
  a period at the end), even though it’s usually a sentence fragment. It should
  not be necessary to list the type, since this is added by the API doc
  generator. Add additional sentences for further description, as appropriate.

- Should usually begin with “The” or “A.”

### Errors

You can use the `Raises` keyword to describe error conditions for a function.
Note that this isn’t currently supported by the Mojo API doc tooling, and will
render as regular text in the function description, not as a separate section.

```plaintext
Raises:
  An error if the named file doesn't exist.
```

### Code examples

Add an `Examples:` header, then include each code sample as a markdown fenced
code block, specifying the language name. The examples section should go after
any other sections (`Parameters:`,`Args:`, `Constraints:` `Returns:`,
`Raises:`).

### Example

```mojo
fn select[
    result_type: DType
](
    self,
    true_case: SIMD[result_type, size],
    false_case: SIMD[result_type, size],
) -> SIMD[result_type, size]:
    """Produces a new vector by selecting values from the input vectors based on
    the current boolean values of this SIMD vector.

    Parameters:
        result_type: The element type of the input and output SIMD vectors.

    Args:
        true_case: The values selected if the positional value is True.
        false_case: The values selected if the positional value is False.

    Returns:
        A new vector of the form
        `[true_case[i] if elem else false_case[i] for i, elem in enumerate(self)]`.

    Examples:

    ```mojo
    v1 = SIMD[DType.bool, 4](0, 1, 0, 1)
    true_case =  SIMD[DType.int32, 4](1, 2, 3, 4)
    false_case = SIMD[DType.int32, 4](0, 0, 0, 0)
    output = v1.select[DType.int32](true_case, false_case)
    print(output)
    ```

    """
```

## Structs/Traits

- Do not repeat the name in the first sentence.

- Use a noun phrase to describe what the type *is* (”An unordered collection of
  items.”).

- Or, similar to function descriptions, use a present tense verb (when possible)
  to describe what an instance does or what the data represents (“Specifies,”
  “Provides,” “Configures,” etc.).

- Optionally include code examples, as with functions.

- Docstrings for traits follow the same rules as docstrings for structs, except
  that traits can't have parameters or fields—only method definitions.

### Example

```mojo
struct RuntimeConfig:
"""Specifies the Inference Engine configuration.

Configuration properties include the number threads, enabling telemetry,
logging level, etc.
"""
```

## Fields or aliases

Be descriptive even when the name seems obvious.

### Example

```mojo
var label: Int
"""The class label ID."""

var score: Float64
"""The prediction score."""
```

### Parameters

Structs can have parameters, which follow the same rules as function parameters.

## Constraints

Mojo functions can have compile-time *constraints,* defined using the
[`constrained()`](https://docs.modular.com/mojo/stdlib/builtin/constrained#constrained)
function. If the constraint isn’t met, compilation fails. Constraints can be
based on anything known at compile time, like a parameter value. You can't
create a constraint on an *argument*, because argument values are only known at
runtime.

Document constraints using the `Constraints` keyword:

### Example

```plaintext
Constraints:
    The system must be x86 and `x.type` must be floating point.
```

If the only constraints are simple limits on single parameters, they should be
documented as part of the parameter description:

Example:

```plaintext
Parameters:
    size: The size of the SIMD vector. Constraints: Must be positive and a
          power of two.
```

For consistency, use the plural “Constraints” even when documenting the
constraint inline in the parameter description. When describing a constraint on
 a single parameter, use a sentence fragment omitting the subject:

```plaintext
# AVOID
    type: The DType of the data. Constraints: This type must be integral.

# PREFER
    type: The DType of the data. Constraints: Must be integral.
```

Always use the standalone “Constraints” keyword if the constraint doesn’t neatly
fit into the description of a single parameter. For example, the constraints on
a struct method may be based on parameters on the struct itself, or on the
machine architecture the code is compiled for.

**Don’t** use the term “constraints” for runtime limitations or error
conditions. Wherever possible, be specific about what happens when a runtime
value is out of range (error, undefined behavior, etc.).

```plaintext
# AVOID
Arguments:
    value: The input value. Constraints: Must be non-negative.

# PREFER
Raises:
    An error if `value` is negative.

# OR

Returns:
    The factorial of `value`. Results are undefined if `value`
    is negative.
```
