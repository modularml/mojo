:title: max
:sidebar_position: 1

.. raw:: markdown

    The `max` command line tool runs and benchmarks MAX pipelines from one
    binary. Use `max serve` to host an OpenAI-compatible endpoint, `max generate`
    or `max encode` to run a model directly, `max benchmark` to load-test a
    running server, `max warm-cache` to compile and cache a model ahead of
    deployment, and `max list` to discover the architectures MAX supports.

    To install the `max` CLI, install the `modular` package as shown
    in the [install guide](/packages#install).


.. click:: max._entrypoints.pipelines:main
  :prog: max
  :hide-description:


.. toctree::
   :hidden:

   benchmark.rst
   encode.rst
   generate.rst
   list.rst
   serve.rst
   warm-cache.rst
   warm-interpreter-cache.rst
