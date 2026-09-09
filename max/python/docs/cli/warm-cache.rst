:title: max warm-cache


Preloads and compiles the model to optimize initialization time by:

- Pre-compiling models before deployment
- Warming up the Hugging Face cache

This command is useful to run before serving a model.

For example, compile and cache a model hosted on Hugging Face:

.. code-block:: bash

    max warm-cache \
      --model google/gemma-3-12b-it

To compile for a target API and architecture without requiring matching
physical hardware, pass ``--target`` (for example, ``cuda``,
``cuda:sm_90``, or ``hip:gfx942``). MAX uses virtual devices for the
compilation, which is useful when building MEF caches on a CI host that
doesn't have the deployment hardware attached:

.. code-block:: bash

    max warm-cache \
      --model google/gemma-3-12b-it \
      --target cuda:sm_90

.. raw:: markdown

    :::note

    The Modular Executable Format (MEF) is platform independent, but
    the serialized cache (MEF files) produced during compilation is
    platform-dependent. This is because:

    - Platform-dependent optimizations happen during compilation.
    - Fallback operations assume a particular runtime environment.

    Weight transformations and hashing during MEF caching can impact performance.
    While efforts to improve this through weight externalization are ongoing,
    compiled MEF files remain platform-specific and are not generally portable.

    :::

.. click:: max._entrypoints.pipelines:cli_warm_cache
  :prog: max warm-cache
  :hide-description:
