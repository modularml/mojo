:title: max warm-interpreter-cache

MAX includes an interpreter that runs operations one at a time as your graph
calls them. For some of these operations, such as matrix multiplication and
elementwise math, the interpreter builds an optimized, compiled version the
first time it runs the operation. It saves each compiled version to an on-disk
cache and reuses it on later runs.

There's one compiled version for each combination of operation, device, and
data type your machine supports, so compiling them all on first use can take
several minutes. Use ``max warm-interpreter-cache`` to compile every combination
for the current hardware up front.

Because the compiled results depend on the hardware, run this command on the
same kind of machine you plan to run on. A common use is during system
provisioning, such as a step in a Dockerfile after you install MAX.

MAX saves the cache next to the engine's own model cache and records your
machine's hardware alongside it, so other MAX processes on the same machine
reuse the compiled results with no extra setup.

The command refuses to run only when a set variable's value conflicts with
the warm: ``MAX_EAGER_ALLOW_LAZY_COMPILE=0`` for a real warm, or
``MAX_EAGER_OP_PRECOMPILE=1`` for either mode; ``--check`` tolerates the
former since it compiles nothing. Unset the variable for this command, for
example ``env -u MAX_EAGER_ALLOW_LAZY_COMPILE max warm-interpreter-cache``.

Use ``--check`` to report whether this machine is already warmed, without
compiling anything:

.. code-block:: console

    $ max warm-interpreter-cache --check

This exits with status 0 if the machine is warmed, or 1 otherwise, so a
provisioning script or health check can branch on the result.

Running the command again on an already-warmed machine does nothing. Pass
``--force`` to recompile anyway, such as after a toolchain change:

.. code-block:: console

    $ max warm-interpreter-cache --force

Compilation runs concurrently in worker processes, one per operation family
by default, capped at the CPU count. Pass ``--jobs`` to bound the number of
workers, or ``--jobs 1`` to compile serially in-process.

.. click:: max._entrypoints.pipelines:cli_warm_interpreter_cache
  :prog: max warm-interpreter-cache
  :hide-description:
