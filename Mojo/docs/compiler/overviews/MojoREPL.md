# Mojo🔥 REPL

## Introduction

A Read Eval Print Loop, or REPL, is an effective tool for providing a powerful
interactive development experience. Mojo provides a powerful REPL experience
built on top of the [LLDB debugger](https://lldb.llvm.org/), which also provides
the debugging environment for the Mojo Language.

See the `MojoLLDB.md` document for additional information for Mojo REPL
developers.

## Getting started

There are several entry points with which to experience the Mojo REPL, with the
main two being the [LLDB REPL](#lldb-command-line-repl), and a
[Jupyter Notebook](#jupyter-notebook). Each entry point contains specific setup
instructions, please refer to each section for more detailed information.

## Important definitions

- Cell: an entry in a notebook which contains code that can be evaluated by the
  Mojo REPL.
- Expression: piece of code that can be evaluated by the Mojo REPL. It can
  originate from a notebook cell or provided by the user via the CLI REPL.

### LLDB Command Line REPL

In addition to providing the underpinning technology, LLDB can also be used as a
command line driver for interacting with the REPL. To start an interactive REPL
session within LLDB, Mojo provides a convenient utility with the necessary
setup:

```shell
# Build the Mojo driver, along with the REPL and all of its dependencies.
build mojo-sdk

# Launch the REPL.
mojo
```

Once run, you'll be provided with a REPL environment where you can immediately
start running expressions, which are delimited by blank lines as in the
following example:

```shell
Welcome to Mojo.
Type :help for assistance.
  1> var my_var = "Welcome to Mojo!"
  2.
  2> print(my_var)
  3.
  Welcome to Mojo!
```

As you can see, the REPL persists variables that are created in an expression so
that they can be accessed in later expressions. The only exception are variables
that start with the `__lldb` prefix, which are discarded right after the
expression. In fact, this prefix is used for internal variables in the
underlying expression evaluation engine.

### Jupyter Notebook

Jupyter notebooks are a common environment for interacting with REPLs of all
shapes and sizes. Mojo provides a custom kernel implementation for interacting
with the REPL in any jupyter environment.

```shell
# Ensure the Mojo Jupyter Kernel is installed in the local environment.
jupyter-init

# Build all of the necessary REPL functionality to run the jupyter kernel.
build MojoJupyter
```

#### VSCode Notebooks

VSCode provides a powerful suite of notebook functionality, which can be easily
integrated with the Mojo Kernel. To change the kernel within a notebook, simply
pick `Select Kernel` in the upper right of the notebook, and select Mojo.
Depending on your setup, you may need to find the kernel via:
`> Select Another Kernel > Jupyter Kernel > Mojo`

#### JupyterLab Notebooks

JupyterLab is the latest web-based interactive development environment for
notebooks provided by the Jupyter Project. The kernel should be available
directly, but you may need to initialize Jupyter first if you haven't already:

```shell
# Setup JupyterLab and the Mojo Jupyter extension.
jupyter-init

# Start a Jupyter server.
jupyter-lab
```

NOTE: On macOS, if you have an ASAN build, this command will not run. You
should use mojo-jupyter-executor for that.

## Features

### `%%python` expressions

The Mojo REPL provides built in support for evaluating full Python expressions,
written natively in Python. To execute a Python expression, simply use
`%%python` as the first line of the expression. Try running the following:

```python
%%python
import sys
print(f"Python version {sys.version}")
```

These Python expressions execute in the same environment as Mojo expressions,
meaning that all of the execution state is accessible via Mojo’s python interop.
This allows for easily composing both Mojo and Python.

As part of executing in the same environment, the Mojo REPL will also
automatically expose information written in python expressions to the Mojo
environment. For example, any variables, functions, or imports defined within a
python expression are directly available for access by future Mojo expressions.

### `%cd`

Change the current working directory.

This command automatically maintains an internal list of directories you visit
during the REPL session. Usage:

- `%cd 'dir'`: change to directory `dir` and push it on the directory stack.
- `%cd -`: pop the directory stack and change to the last visited directory.

### `%#`

Hide a source code line from generated documentation.

When used within a doc string code block, the code line after the directive
will be hidden from any generated documentation. For example:

```mojo
var value = 5
%# print(value)
```

will generate documentation of the form:

```mojo
var value = 5
```

Hidden lines are processed as if they were normal code lines during execution.

## Limitations and Sharp Edges

Mojo is still young and growing fast, and this also applies to the REPL
environment. We aim to provide a powerful REPL experience that developers have
come to expect, but we’re still building there one step at a time. Below are a
few caveats and limitations of the current REPL environment, many of which will
be improved as we develop Mojo.

### Unable to redefine implicit variables

Mojo provides support for defining implicit variables. These variables are
defined by assigning to a name, not by using the `var`:

```mojo
def foo():
  # Here we've defined a new variable named `a`.
  a = 10
```

The REPL is currently unable to discern when an implicit variable is being
assigned to a new value, and when a new variable was intended to be defined. For
example, consider the following expression:

```mojo
struct S:
  var value: Int

  def __init__(out self, x : Int):
    self.value = x

s = S(10)
print(s.value)
```

Consider we re-execute this expression after adding a new field to S:

```mojo
struct S:
  var value: Int
  var value2: Int

  def __init__(out self, x : Int):
    self.value = x
    self.value2 = 15

s = S(10)
print(s.value)
```

When executing this second expression, the REPL will misidentify the intention
that `s = S(10)` is a new variable using the updated S type, and emit an error
about a missing conversion from the S type from the new expression and the S
type from the old expression.

**Workaround:**

This issue only applies to implicit variables. Those defined with
`var` may be freely redefined as many times as desired. If an implicit variable
needs to be overwritten, consider using `var` to introduce the variable
for now.

### Variable lifetimes behave unexpectedly

Variables defined in the “top scope” of the REPL currently behave slightly
differently than those defined within a `def` or `fn`, meaning that the powerful
ownership modeling provided by Mojo may behave unexpectedly and/or result in
errors associated with these variables.

**Workaround:**

This behavior only applies to top-level variables. Variables defined within a
`def` or `fn` should behave exactly as expected, and can be used to fully play
around with the Mojo’s powerful ownership model as intended.

### REPL environment defaults to `def`

The REPL environment runs all expressions within a `def` environment, not an
`fn` one. This shouldn't have much if any significance for most user code, but
is a potential foot-gun that devs should be aware of.

## Debugging Compiler Issues

Debugging compiler issues within the REPL environment is much different from
debugging issues with a single .mojo module. Given that the REPL executes
expressions across multiple invocations, it requires a different kind of mindset
when debugging a crash or miscompilation. This section contains useful tips and
tricks to make debugging issues within the REPL a bit easier.

### mojo-jupyter-executor

Jupyter notebooks generally involve a UI frontend component, with the backend
execution somewhat hidden and difficult to interact with outside of logs; not to
mention that the backend entry point is defined within python. This makes it
difficult to debug Mojo issues the traditional way, i.e. via a debugger. To make
this debugging flow a bit easier, Mojo provides a utility
`mojo-jupyter-executor` that can be used to execute a notebook in an environment
that is amenable to traditional debugging, e.g. via LLDB. To execute a notebook,
simply provide it to `mojo-jupyter-executor`. This will launch the Mojo Jupyter
kernel and execute each cell individually, as you would expect in a normal
jupyter environment.

```shell
mojo-jupyter-executor notebook.ipynb
```

The executor also has a REPL mode, where you can execute an individual cell at a
time. You can start the executor in this mode by running:

```shell
mojo-jupyter-executor
```

You will see a command prompt, where you can run simple commands like so:

```shell
[0] > print("hello")
[stdout] hello

[0] > :next-cell
[1] >
```

The number in square brackets is the 'cell ID' - this does not auto-increment
because you might want to (for example) dump the logs from a previous command in
the same cell. You can control which cell you're in with the special commands
`:next-cell` and `:prev-cell`. These increment and decrement the cell counter,
respectively. You can also cleanly exit the REPL mode by running `:exit`.

If you are in REPL mode and you'd like to execute a notebook, you can run
`:notebook /path/to/notebook`. This mode behaves as if the `--debug-on-failure`
flag is set, and drops you into debug mode if a cell fails.

You can also execute LLDB commands in the jupyter executor the same way you
would in the REPL - by running `:<command>`. For debugging a crash, it's often
useful to run:

```shell
[0] > :settings set target.process.unwind-on-error-in-expressions false
[0] > :settings set target.process.ignore-breakpoints-in-expressions false
```

These two commands allow you to set breakpoints within the code running in your
expressions, and inspect the thread state at the point of failure.

### `--debug-on-failure`

When launching the `mojo-jupyter-executor` with a notebook, you can specify
`--debug-on-failure` and the executor will automatically drop you into REPL
mode if a notebook cell fails. We set `unwind-on-error-in-expressions` to
`false` by default, so this should allow you to trace up the stack in case of
a crash by running LLDB expressions.

### Future Projects

Empty for now, please add ideas here!

## jupyter-cli-executor

The `jupyter-cli-executor` is another tool that can be used to execute Mojo or
Python notebooks in the CLI. Unlike `mojo-jupyter-executor`, it uses a Jupyter
server, thus mimicking the Jupyter UI environment as closely as possible.
However, it is very difficult to debug and might be flaky in the CI due to
limitations in parallel executions, therefore it should only be preferred over
`mojo-jupyter-executor` if going through the Jupyter server is a requirement.

## `MojoREPL` Developer Guide

### Logging

You can log information by using the methods on `MojoExpressionLogger`. The way
we log is by emitting events to the LLDB event handler interface.

The way we treat event kinds is as follows (list in `MojoExpressionLogger.h`):

- `BroadcastUserMessage` events are flushed to the user's stderr immediately.
- `DebugLog` events are only flushed when the `expr` LLDB log channel is
  enabled via `:log enable lldb expr`.
- `DumpIR` events are only flushed when the `expr` LLDB log channel is enabled
  in verbose mode via `:log enable lldb expr -v`.
- `ErrorLog` events are flushed immediately.

Events are mostly handled by `MojoExpressionLogger::handleEvent`, which
implements the behavior above. A new user can do whatever they want with the
various events as befits their specific application.

Feel free to add more event kinds as is appropriate - event kinds ending with
`Message` are shown to the user in the notebook, while event kinds ending in
`Log` are not shown to the user.

Lastly, as a troubleshooting aid you can pass the `-f /tmp/logs.txt` option to
the `:log enable lldb expr` command to output the logs to a file for easier
inspection.

### LLDB and Mojo Commands

We support executing arbitrary LLDB commands in the Notebook and the CLI REPL.
If an expression starts with `:`, then the rest of the text is handled as an
LLDB command.

Moreover, we support a set of Mojo commands inside the LLDB command tree, which
do things like dumping internal logs. Again, feel free to add new commands, just
please document them here!

Current Mojo commands:

- `:mojo help repl` - This command prints out a REPL help text.

### Testing

We support llvm-lit tests in our CI for both the REPL and the jupyter
environments. They are located in `KGEN/test/mojo-repl` and
`KGEN/test/mojo-jupyter` respectively.

As a trick, when debugging llvm-lit test issues, you can produce helpful logs
by including the expression `:log enable lldb expr -f /tmp/logs.txt` directly in
the test file in the case of a REPL test, or as a new cell in a jupyter test.
This won't affect the data stream received by FileCheck.
