# Mojo docstring style guide

This guide sets the style rules for docstrings across the Mojo standard
library. Docstrings are written in Markdown, not reStructuredText.

## Why docstrings matter

Code is read more often than it's written. Docstrings are for developers,
contributors, maintainers, and every tool that consumes our source.

A good docstring makes intent obvious. A great one prevents bugs, misuse,
and redesign.

### Philosophy

**Docstrings are part of the API**. They create the public surface of the
language. Writing a docstring reveals unclear naming, missing constraints,
or surprising behavior.

**Optimize for humans**. Assume the reader is smart, busy, and skimming.
Avoid cleverness, jokes, filler, and insider language that might not be
understood by a global audience. Prefer simple words, short lines, and
concrete statements. However, when in doubt, prefer clarity over brevity.

**Consistency matters**. A consistent style lets readers move faster, trust
what they read, and compare APIs easily. Follow this guide even when you
think you don't need to.

**Docstrings are read out of context**. Assume the reader arrived through a
web search or an IDE hover. Every docstring must stand alone. Don't rely on
surrounding code or adjacent symbols to carry meaning.

### Four consumers

Every docstring serves four audiences simultaneously:

- **Human developers** read them in web docs, search results, and IDE hover
  panels. They need clear prose, concrete examples, and honest descriptions
  of edge cases.

- **Machine translation engines** convert documentation into other languages.
  They need simple, direct sentences and consistent terminology. A single
  ambiguous sentence becomes ambiguous in every language.

- **LLMs and AI assistants** ingest docstrings when answering questions and
  generating code. They depend on consistent structure, complete sections,
  and correct vocabulary. A missing `Raises:` section or a vague `Returns:`
  description becomes a hallucination risk.

- **IDEs** surface the summary line as hover text and structured sections as
  autocomplete hints. The summary must be self-sufficient. It's often the
  only line a developer sees.

## Structure

Every docstring has the same fundamental shape, whatever it documents:

- A one-sentence summary that ends with a period.
- Optional body paragraphs, separated from the summary by a blank line.
- Optional named sections, such as `Args:`, `Parameters:`, and `Returns:`.

The summary is mandatory. The body and named sections appear when the
declaration needs them. [Docstring components](#docstring-components)
describes each part in detail.

The `mojo doc --diagnose-missing-doc-strings` command validates public API
coverage. By default, it reports each missing or partial docstring as a
*warning*, still generates the docs, and exits successfully. Adding
`-Werror` promotes those warnings to errors: `mojo doc` prints
`could not generate documentation`, produces no output, and exits with a
non-zero status.

Coverage is enforced when you build the standard library, not as a
`mojo doc` default. The build always runs `mojo doc` with `-Werror`, and
the `std` docs target turns on the missing-docstring check, so any missing
or incomplete docstring fails the build. Run the same check locally before
you push:

```bash
mojo doc --diagnose-missing-doc-strings -Werror -o /dev/null stdlib/std/
```

### Public symbols must have a docstring

Mojo's check requires a docstring on:

- Public functions and methods (names not starting with `_`), plus dunder
  methods like `__init__()` even though their name starts with `_`
- Public structs and traits
- Public modules. A package's `__init__.mojo` carries the package's
  docstring.
- Public struct fields
- Public parametric aliases (`comptime Name[...] = ...`)

### Exempted symbols

The `docs` check exempts a symbol from documentation when any of these
conditions apply:

- Its name starts with `_` but isn't a dunder. For example, `_helper()` is
  exempt, but `__init__()` isn't. This covers private implementation details.
- It's defined in a private module or package. The check skips the entire
  module or package.
- It uses the `@doc_hidden` decorator.
- The compiler synthesized it. For example, a member method that shares its
  parent struct's source location.
- It's a nested function or closure. The check only considers module-,
  struct-, and trait-level symbols.

Infer-only parameters (the `[T, //, ...]` form) aren't checked by the
automated coverage tool, but document them anyway per the rule below.

### Every docstring must cover its signature

Once a public symbol has a docstring, the automated check requires it to
document every relevant part of its signature:

- A function with parameters needs a `Parameters:` section.
- A function with arguments needs an `Args:` section.
- A function that returns a value needs a `Returns:` section.
- A function that can raise needs a `Raises:` section.
- A struct or parametric alias with parameters needs a `Parameters:` section.

The automated check doesn't require other sections. `Preconditions:`,
`Constraints:`, and `Safety:` are required by this guide when their
triggering condition applies (see the table in
[Named sections](#named-sections)), but
`mojo doc --diagnose-missing-doc-strings` doesn't check for them.

## Docstring components

A docstring is built from a summary line, an optional body, and named
sections, in that order.

### Summary line

Every docstring summary introduces packages, modules, and declarations.
They are one-line items, and tools often use them to build documentation
structure. They need to be concise and follow a set pattern:

- Begins with a capital letter or a backticked term.
- Doesn't repeat the name of the symbol being documented.
- Starts with a subjectless present-tense verb phrase. Never open with
  "This method," "This function," or "This struct."
- Ends with a period.

### Docstring body

Bodies follow the summary, separated by a blank line. Named sections
follow in turn.

A docstring divides its work in two. The named sections—`Parameters:`,
`Args:`, `Returns:`, `Raises:`, and the others—document the signature in
a fixed, labeled format that readers and tools can scan. The body is
free-form prose that covers what those sections can't: the API's meaning,
the guarantees it makes, and how and when to use it correctly.

The body continues the summary with the information a reader needs to use
the API correctly. Prefer prose over creating new titled sections.

Every body:

- Uses standard English, with its normal casing and punctuation rules.
- Uses blank lines between paragraphs.
- Aligns code fences with the surrounding text.

Use the body for:

- Usage guidance.
- Non-obvious behavior.
- Edge cases and caveats.
- Comparisons with closely related APIs.
- Equivalence relationships.
- Implementation rationale that affects observable behavior.
- Short examples that reinforce the discussion.

If you're thinking of adding `Notes:`, the content belongs in the body
instead.

**Style guidelines**:

- Avoid bold text. Use it only for the occasional critical point.
- Italicize the first use of a new concept.
- Use Markdown links for external references, such as standards documents
  and research papers.
- Avoid hedging and throat-clearing language. Keep the body direct,
  contractual, and concise.

**Yes**:

```text
If you need an owned optional value rather than a reference, use
`Optional`.
```

```text
For bounds-checked access to a contiguous range, see `Span`.
`Pointer` skips those checks and is intended for
performance-critical code where safety is guaranteed by the caller.
```

**No**:

```text
Note that this is similar to Optional but different.
```

Why not: It opens with "Note that" and never explains the actual
distinction.

### Named sections

Named sections follow the body. They document parts of a declaration's
contract that are visible in its signature or enforced by the compiler or
runtime.

Include every section that applies. Don't omit required sections or add
sections that don't apply.

<!-- markdownlint-disable MD013 -->

| Context                                                             | Required section |
|---------------------------------------------------------------------|------------------|
| Function/method has compile-time parameters                         | `Parameters:`    |
| Function/method has runtime arguments                               | `Args:`          |
| Function/method has a return value                                  | `Returns:`       |
| Function/method can raise                                           | `Raises:`        |
| Struct has compile-time parameters                                  | `Parameters:`    |
| Parametric alias has parameters                                     | `Parameters:`    |
| Declaration has precondition (on caller), runtime assert            | `Preconditions:` |
| Declaration has compile-time constraints `where`, `comptime assert` | `Constraints:`   |
| Function/method is unsafe: violating a caller obligation causes UB  | `Safety:`        |

<!-- markdownlint-enable MD013 -->

A single-parameter `Constraints:` requirement may be satisfied inline
within `Parameters:` instead of as a standalone section—see
[`Constraints:`](#constraints) below for when a standalone section is
needed instead.

Each section has its own content and formatting rules, described in the
sections that follow.

### `Parameters:` and `Args:`

- One entry per line: `name: Description ending with a period.`
- For defaulted arguments and parameters, state the default inline: `start: The
  starting index (defaults to 0).`
- For `NoneType` parameters: `` value: `None`. ``
- Continuation lines indent 4 spaces.

### `Returns:`

- Full sentence ending with a period.
- Prefer active voice: "A reference to the contained value." Not "The value
  is returned."
- For reference-returning methods: `A reference to the contained value.`
- For `Maybe`-returning methods: "A `Maybe` holding a reference to the
  first element, or an empty `Maybe` if the list is empty."

### `Raises:`

- `ErrorType: Condition as a sentence fragment ending with a period.`
- Document every known or specified error type. A bare `Raises:` with no
  named types isn't sufficient.

**Yes**:

```text
Raises:
    FileNotFoundError: If the named file does not exist.
    PermissionError: If the caller does not have read access.
```

**No**:

```text
Raises:
    If something goes wrong.
```

### `Preconditions:`

Documents a runtime assertion enforced on the caller. A failed
precondition aborts execution, and you can't catch it with try/except,
unlike a raised error.

**Yes**:

```text
Preconditions:
    The list must not be empty.
```

Don't use `Preconditions:` for conditions that raise a catchable error.
Those belong in `Raises:`.

`Preconditions:` is always plural, even for a single condition. This avoids
renaming the section if you add more preconditions later.

### `Constraints:`

Documents the contract a caller must follow to successfully invoke or
construct the declaration: compile-time constraints that cause
compilation to fail if unmet. When the valid values are known, state
them.

A `where` clause or `comptime assert` states the mechanism, not always
the full reasoning behind it. State that reasoning here.

**Yes**:

```text
Constraints:
    Must be a power of 2; valid values are 8, 16, 32, 64.
```

Simple constraints on a single parameter may appear inline in
`Parameters:` rather than in a standalone section:

```text
Parameters:
    size: The capacity. Must be a power of 2; valid values are 8, 16, 32, 64.
```

Use the standalone section when the constraint spans multiple parameters,
depends on the struct's own parameters, or relates to the compilation
target.

Don't use `Constraints:` for runtime conditions. A runtime assertion
belongs in `Preconditions:`; a catchable error belongs in `Raises:`.

### `Safety:`

Documents an unsafe function or method: what the caller must guarantee,
and the undefined behavior that results from violating it. Unlike a
precondition or a constraint, a safety violation isn't checked at
compile time or runtime—the API trusts the caller.

Only add `Safety:` where misuse causes undefined behavior. Most
functions and methods don't need it.

**Yes**:

```text
Safety:

- The returned memory is uninitialized; reading before writing is
  undefined.
- The returned pointer has an empty mutable origin; you must call
  `free()` to release it.
```

Don't use `Safety:` for conditions the runtime already checks. A checked
condition belongs in `Preconditions:`.

### `Performance:`

Covers time complexity, space complexity, memory allocation behavior, and
any other cost a caller should know about. Optional, but include it when
the cost is non-obvious or when a caller might make a wrong assumption
about performance.

**Yes**:

```text
Performance:
    O(1).
```

```text
Performance:
    Time: O(n).
    Space: O(1):  operates in place with no additional allocation.
```

### `See:`

When you base the implementation on a named algorithm or data structure,
link to it. Don't assume the reader knows the algorithm or will look it
up independently.

```text
See:
    [Swiss Tables](https://abseil.io/about/design/swisstables) for the
    algorithm behind this implementation.
```

### `Examples:`

Show syntax in context:

- Always the last section.
- Named either `Examples:` (preferred) or `Example:`. Existing code uses
  both. Prefer `Examples:` for new docstrings and when updating existing
  ones.
- Fenced ` ```mojo ` block, no trailing whitespace.
- The code fence block is left-aligned with `Examples:`, with a space
  between the section name and the code block. This differs from every other
  section's content.
- Variables have concrete, readable names: `numbers`, `names`, `fruits`,
  `index`, not `lst` or `x`.
- Use `var` and `ref` consistently for variable binding sites.
- Expected output shown as `# value` inline comment on the producing
  line. Avoid `Output:` followed by a code block where possible.
- Show the primary usage pattern plus any non-obvious pattern unique to
  the method.
- Don't show every overload: pick the most instructive one.

The appropriate scope depends on the level:

- **Packages and modules**: the conceptual entry point, not a tour of
  individual APIs.
- **Types and traits**: construction and the primary access pattern.
- **Methods and functions**: the call in context: normal case, non-obvious
  case, and gotchas.

### Undocumented section titles

Mojo's tooling supports arbitrary section titles. We ask that you don't add
new sections without going through Mojo language design approval with the
docs team looped in.

We ask that new conventions be limited to those that won't require future
edits, can't be expressed in the docstring body or an existing section, and
pass the undercaffeinated engineer test: can a reader skimming the
docstring immediately understand when the section applies, and can an API
author immediately understand what belongs there?

### Section order

Place validated sections first, ad-hoc sections after:

Validated: `Parameters:` → `Args:` → `Returns:` → `Raises:` →
`Preconditions:` → `Constraints:` → `Safety:`

Ad-hoc: `Performance:` → `See:` → `Examples:`

The compiler doesn't enforce ordering among sections, but consistency
helps readers scan.

Duplicate sections, empty sections, and over-indented section tags all warn,
same as the coverage diagnostics above (`-Werror` promotes each to an error).

### Dependent parameter and argument rules

- Documented name must exactly match the declared name.
- Each name may appear only once.
- Documented order must match declaration order.
- Every entry must have a description.
- Each description must start with a capital letter and end with a period.
- Document infer-only parameters (before a `//` marker) clearly as
  inferred. They may not appear at callsites.
- Do not document `out` arguments. From the caller's perspective,
  they aren't visible. They are the implementation detail that provides
  alternate behavior. Document them in `Returns:` if needed.

### Section formatting rules

- Section tags (`Args:`, `Returns:`, etc.) must be at the standard indent
  level.
- Each section name may appear only once.
- A section with a header but no content warns, same as above.

**Unexpected sections produce warnings**:

- `Returns:` on a function with no return value.
- `Raises:` on a function that cannot throw.

## Formatting

`mojo format` is the canonical formatter. Whatever it produces is correct,
by definition. The formatting it enforces is a hard requirement, not a
style suggestion.

For docstrings, `mojo format` applies these formatting rules:

- Rewrites `'''...'''` as `"""..."""`.
- Normalizes string prefixes. For example, `R` becomes `r`, it removes
  redundant `u` prefixes, and it reorders multi-character prefixes
  consistently (`fr"""..."""` becomes `rf"""..."""`). It preserves `f`- and
  `t`-string prefixes.
- Reindents the docstring body, removes leading and trailing blank lines,
  and strips trailing whitespace.
- Adds spacing when needed to prevent the docstring text from merging with
  or escaping the closing quotes.
- Keeps the closing `"""` on the last line when it fits within `mojo
  format`'s 80-column limit. Otherwise, moves it to its own indented line.
- Leaves docstrings containing backslash-newline escapes unchanged because
  reformatting would change their meaning.

`mojo format` enforces an 80-column line length as a ceiling it won't
exceed for code, but it doesn't reflow prose inside docstrings. Our
in-house style, described next, holds docstring prose to that same
80-column limit, applied manually.

## In-house style

Although Mojo is not Python, our docstrings are essentially Pythonic. In
addition to this document, we rely on the following standards:

- Primary standard: [PEP-0257](https://peps.python.org/pep-0257/).

- Secondary standard: Google's [Python docstring
style](https://google.github.io/styleguide/pyguide.html).

- Additional reference: [PEP-0008](https://peps.python.org/pep-0008/).

When in doubt, fall back to these three, in the order listed. This
document takes precedence over all three when they conflict with it.

### Wrapping

Wrap docstring lines at 80 columns—the same limit `mojo format`
enforces on code. Because `mojo format` doesn't reflow prose inside
docstrings, wrap docstring text to 80 columns manually before running the
formatter.

We use 80 columns rather than PEP 257's 72-column recommendation for
docstrings. The tighter limit forces awkward wrapping once you account for
indentation, and it's worse for docstrings nested inside structs and
methods. Matching `mojo format`'s 80-column ceiling also keeps a single,
consistent limit across code and docstrings.

The count includes indentation, punctuation, and triple quotes. For
one-line docstrings, it includes both the opening and closing triple
quotes.

Inside fenced code examples, keep lines to 80 columns where it reads
naturally, but correctness and clarity of the example come first—don't
break an expression across lines just to fit. Inline output comments
(`# value`) that trail past 80 columns are acceptable. `mojo format`
doesn't reflow code inside docstrings, so these lines are never adjusted
automatically.

### Naming and backticking

Use backticks to name Mojo keywords, APIs, and bindings inline:

```text
...type and struct reflection via `reflect[T]`, a `comptime` alias...
```

Backticking is conventional for argument and parameter names:

```text
`T` must be a struct type.
`idx` must be in range `[0, field_count())`.
```

Use `name()` or `name[]()` to name other functions in text. For example:

```text
Use `copy()` to create an independent copy of the value.

Use it as `reflect[T].method()` rather than constructing an instance.
```

To identify a specific overload, add argument names or a parameter list:

```text
Use the `reversed(value: List[T])` overload to reverse a `List`.
```

*For C FFI only*, omit the parentheses because dynamic libraries look these
up by text name. The example here uses a comment, but the same rule applies
to docstrings:

```mojo
# Get the libc `strlen` function from the process handle
var c_strlen = proc.get_function[
    def(Pointer[c_char, line_origin])
    thin abi("C") -> c_size_t
]("strlen")
```

### Math

We use KaTeX for mathematical notation.

- Use `$$...$$` for display equations. Avoid inline math (`$...$`).
- Double-escape backslashes because KaTeX appears inside a Mojo string.
  For example, write `\\mathcal`, `\\Theta`, and `\\times`.
- Keep equations focused. Explain variables and assumptions in the
  surrounding prose rather than inside the equation.
- To write a literal `$$` outside an equation, escape it as
  `<span>$$</span>`. KaTeX doesn't render `$$` inside backticks or a code
  block, so no escape is needed there.

Common use-cases include algorithm complexity and mathematical definitions:

```text
$$
O(n \\log n)
$$
```

```text
$$
\\Theta(n^2)
$$
```

```text
$$
\\sum_{i=0}^{n-1} i
$$
```

```text
$$
x_{i+1} = x_i + \\Delta t
$$
```

### String escape sequences

Docstrings are Mojo strings, so string escapes apply inside them:

- `\n` renders as a newline.
- `\\` renders as a single literal backslash.
- `\t` renders as a tab.

Use these escapes in examples that show string content containing them,
rather than describing the escape in prose.

## Declaration levels

Different declarations answer different questions. Packages orient.
Modules explain grouping. Types define guarantees. Methods describe
behavior. Don't force them into the same shape.

### Packages

A package docstring is a front door. It answers: what is this for, why
does it exist, and when should I reach for it?

A package docstring orients the reader. It doesn't list everything inside.
Put usage documentation in the module docstring next to the relevant code,
not in the package's `__init__.mojo`.

**Formatting**:

- Starts with a subjectless present-tense verb phrase. Never open with
  "This method," "This function," or "This struct."
- Avoid the package name or refer to "this package."
- After a blank line: 1–2 short paragraphs explaining why the package
  exists and when to use it.
- No symbol inventories: they age fast and create churn.
- Avoid examples unless they explain the package's conceptual purpose
  and can't be expressed in prose alone.

**Yes**:

```mojo
"""Provides safe and unsafe memory access primitives.

The pointer types in this package cover the range from reference-counted
ownership to raw address manipulation. Reach for `OwnedPointer` when safety
matters and `Pointer` when you need direct control and can guarantee
correctness at the call site.
"""
```

**No**:

```mojo
"""The memory package.

This package contains Pointer, AddressSpace, and other
memory-related types. It also includes OwnedPointer and Arc for
reference-counted ownership.
"""
```

Why not: repeats the package name, inventories symbols, doesn't explain
when to use it or why it exists.

### Modules

A module docstring explains the cohesive idea behind a group of APIs. It
describes the shared purpose and design boundaries and why these things
belong together.

**Formatting**:

- Summary: Present-tense verb phrase, ends with a period. Don't repeat the
  module name.
- After a blank line: one short paragraph explaining shared purpose and
  how the APIs relate.
- Named patterns use italic: *Check then manifest*: followed by an
  explanation on the same line.

**Yes**:

```mojo
"""Defines optional reference types for values that may or may not exist.

All types in this module represent the same concept: a reference that
might be empty. They differ in ownership semantics. `Maybe` borrows,
`Optional` owns. Choose based on whether the caller needs an independent
copy of the value.
"""
```

**No**:

```mojo
"""This module defines Maybe and Optional.

Maybe is a type for optional references. Optional is a type for optional
owned values. MaybeIter is an iterator for Maybe.
"""
```

Why not: opens with "This module," inventories symbols without explaining
the relationship between them.

### Types and traits

Type and trait docstrings describe what they represent, what they promise,
and how they are meant to be used. This is where behavior, constraints,
and invariants belong.

**Formatting**:

- Summary: present-tense verb phrase, no subject. Don't repeat the type name.
- `Parameters:` section for compile-time parameters. Each entry:
  `T: The type of the referenced value.`
  Inline caveats when the constraint is non-obvious:
  `` T: The element type. Constrained to `Copyable`. ``
- After `Parameters:`, one prose paragraph explaining design rationale
  or key distinction from similar types.
- `Examples:` section last. Highly recommended: at least one example
  showing construction and the primary access pattern.
- Private/internal structs (for example, `_MaybeIter`) follow the same
  rules.
- Use the docstring body (after the one-line summary) for implementation
  details callers need.

**Yes**:

````mojo
"""Models a reference that may or may not be present.

Parameters:
    T: The type of the referenced value.
    origin: The origin of the referenced value.

Unlike `Optional`, which owns its value, `Maybe` borrows through a
reference. Use `Maybe` when you need to express absence without taking
ownership.

Examples:

```mojo
var items: List[Int] = [10, 20, 30]
var first = items.maybe_first()
if first:
    print(first.value())  # 10
```
````

**No**:

```mojo
"""Maybe type.

This is an optional reference type. It can hold a reference or be empty.
Use manifest() to get the value.
"""
```

Why not: summary repeats the type name, doesn't state the guarantee,
doesn't describe the relationship to `Optional`, no parameters, no
examples.

### Required `comptime` members

A trait can declare a `comptime` member with no initializer to require
conforming types to supply their own compile-time value—often an
associated type. Document what the member represents and, when it's a
type, what role it plays for conforming types.

**Yes**:

```mojo
comptime ReversedType: Iterator
"""The iterator type returned by `__reversed__()`."""
```

A `comptime` member with an initializer is a provided value, not a
requirement—document it as an alias instead; see [Compile-time
aliases and parametric aliases](#compile-time-aliases-and-parametric-aliases).

### Fields

Public struct fields need a docstring even when the name seems obvious. The
docstring states what the field represents and any invariant the struct
maintains around it.

**Formatting**:

- Present-tense verb phrase or noun phrase, ends with a period.
- Don't repeat the field name.
- Place the docstring after the `var` declaration, not before it.

**Yes**:

```mojo
var label: Int
"""The numeric identifier assigned when the entry was created."""
```

### Compile-time aliases and parametric aliases

Alias docstrings explain the meaning of the alias, not the expression it
expands to. Parameterized aliases also describe their compile-time
parameters and any non-obvious constraints.

**Formatting**:

- Summary: Present-tense verb phrase, no subject, ends with a period.
- Trivial aliases usually need only a one-line summary.
- Parameterized aliases include a `Parameters:` section.
- Use the docstring body to explain non-obvious type-level constraints or
  design rationale.
- Place the docstring after the `comptime` declaration, not before it.

**Yes**:

````mojo
comptime Element = Self.T
"""Defines the element type of this iterator."""

comptime reflect_fn[func_type: AnyType, //, func: func_type] = ...
"""Reflects a function at compile time.

Parameters:
    func_type: The function's type.
    func: The function to reflect.
"""
````

### Methods and functions

Method- and function-level docstrings are the most concrete. They explain
what the call does, what it expects, what it returns, and what can go
wrong.

Examples are most valuable at this level because they are closest to the
behavior they demonstrate. Show the normal case, the non-obvious case,
and the gotchas.

**Formatting**:

- Present-tense verb phrase, no subject, ends with a period. Don't repeat
  the function name.
- Section order: `Parameters:` → `Args:` → `Returns:` → `Raises:` →
  `Preconditions:` → `Constraints:` → `Safety:` → `Performance:` → `See:` →
  `Examples:`
- Include only sections that apply.
- Never write `Returns: None.`
- All content lines end with a period.

**Yes**:

````mojo
"""Returns the referenced value, aborting if empty.

Check emptiness with `if maybe:` before calling. An unconditional
call to `manifest()` on an empty `Maybe` aborts the program.

Args:
    self: The maybe reference.

Returns:
    A reference to the contained value.

Raises:
    EmptyMaybeError: If this `Maybe` is empty.

Performance:
    O(1).

Examples:

```mojo
var items: List[Int] = [10, 20, 30]
var first = items.maybe_first()
if first:
    print(first.manifest())  # 10
```

"""
````

**No**:

```mojo
"""This method returns the value.

Returns:
    The value.
"""
```

Why not: opens with "This method," summary repeats the name concept,
`Returns:` is vague, missing `Raises:` for a method that can fail,
no complexity, no examples.

## Voice and language

Docstring voice is precise and opinionated, not conversational. Write as if
speaking to a capable peer. It's a contract reference, not engagement: it
describes what something is and what it guarantees, and trusts the reader to
take it from there.

Two rules govern everything below:

- State the contract.
- Omit everything that doesn't help the reader use the API correctly.

The voice has four characteristics:

- **Subjectless, present-tense, declarative.** "Returns the referenced value,
  aborting if empty." No subject and no "this function"—start with the verb,
  in the present tense ("expects," "returns," "raises").

- **States the guarantee and the rationale, not the mechanism.** Honest about
  non-guarantees too: "Do not rely on `size_of()`. Treat the result as a
  non-stable implementation detail."

- **Second person only in guidance, quarantined to the docstring body.** The
  description never addresses the reader; the advice does. Use second person
  and imperatives for guidance ("Use `manifest()` when...", "Check whether a
  `Maybe` is empty before calling `manifest()`"), and keep it out of the
  summary line.

- **Deliberately flat.** No momentum, no delight, no hook. All four consumers
  read the same line, and personality becomes ambiguity in translation. Write
  one idea per sentence; compound sentences lose meaning in translation.

Analogy, narrative, and encouragement are out of bounds. The voice just
states the contract.

### Clarity

- Use active voice where practical.
  - Avoid: "The task is blocked until the policy allows it."
  - Use: "Waits until the policy allows the task to continue."
- Avoid ambiguous pronouns. Replace `it` and `this` with the actual noun
  when the referent is not the immediately preceding noun.
- Avoid comparisons to other languages or APIs. State what this API does
  directly.
- State what is true today. Avoid "coming soon," "not yet supported," or
  "will be added."
- Include articles where English requires them: "expects a hex string,"
  not "expects hex string."

### Tone

- Avoid throat-clearing and hedging language. Don't begin with
  "Note that" or "It should be noted."
- Don't use "simply," "easily," "obviously," or "just."
- Avoid idioms, metaphors, and culturally specific expressions.
- Don't use humor.

### Terminology and style

- Use American English: `color` not `colour`, `initialize` not `initialise`.
- Write out Latin abbreviations. Use "for example" instead of `e.g.` and
  "that is" instead of `i.e.`.
- Don't add `Returns:` for functions that return no value.

### Canonical terminology

The following table lists the preferred term for each concept and the
terms to avoid.

<!-- markdownlint-disable MD013 -->

| Use        | Not                                                                         |
|------------|-----------------------------------------------------------------------------|
| argument   | param, input (for run-time values)                                          |
| parameter  | arg (for compile-time values)                                               |
| struct     | type, class (at source level)                                               |
| trait      | protocol, interface                                                         |
| conform to | implement, satisfy, inherit                                                 |
| refine     | extend, inherit, subclass (for trait-to-trait relationships)                |
| type       | `DType` (except for the numeric element-type API, where `DType` is correct) |

<!-- markdownlint-enable MD013 -->
