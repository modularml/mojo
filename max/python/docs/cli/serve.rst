:title: max serve

Launches a model server with an OpenAI-compatible endpoint. Specify the
model as a Hugging Face model ID or a local path.

For example, start a server for a Gemma 3 model on the first GPU:

.. code-block:: bash

    max serve \
      --model google/gemma-3-12b-it \
      --devices gpu:0 \
      --max-batch-size 8 \
      --device-memory-utilization 0.9

The endpoints exposed depend on the value of the ``MAX_SERVE_API_TYPES``
environment variable (default: ``openai,sagemaker``):

- ``openai``: ``/v1/completions``, ``/v1/chat/completions``,
  ``/v1/embeddings``, ``/v1/models``, ``/v1/health``
- ``sagemaker``: SageMaker-compatible inference endpoints
- ``kserve``: KServe-compatible inference endpoints
- ``responses``: ``/v1/responses`` (required for the ``pixel_generation``
  task)

The OpenAI routes are always registered when the ``openai`` API type is
enabled, but each only functions when the model is served with a
compatible ``--task`` value (for example ``embeddings_generation`` for
``/v1/embeddings``).

To run inference without an HTTP server, see
`max generate </cli/generate>`_ (text completion) or
`max encode </cli/encode>`_ (embeddings).

For details about the endpoint APIs provided by the server, see [the MAX REST
API reference](/rest-api/).

Running on multiple GPUs
------------------------

Pass a comma-separated list of GPU IDs to ``--devices``:

.. code-block:: bash

    max serve \
      --model google/gemma-3-12b-it \
      --devices=gpu:0,1,2,3 \
      --max-batch-size 16

Use ``--devices=gpu:all`` to target every visible GPU. Omit ``--devices`` to
use the model or config default.

``--devices`` is the first-class device selector for ``max serve``.
Avoid combining it with the shell-level ``CUDA_VISIBLE_DEVICES``
environment variable — the two are translated independently and
stacking them can produce wrong device routing under multi-process
workspaces.

You can extend MAX with your own model implementations by loading custom
architectures through the ``--custom-architectures`` flag. Each value takes
the form ``path/to/module:module_name``:

.. code-block:: bash

    max serve \
      --model google/gemma-3-12b-it \
      --custom-architectures path/to/module1:module1 \
      --custom-architectures path/to/module2:module2


.. raw:: markdown

    :::note Custom architectures

    For a full example using the
    [`--custom-architectures`](#--custom-architectures-custom_architectures) flag,
    see [Serve custom model architectures
    ](/develop/serve-custom-model-architectures).

    :::

.. click:: max._entrypoints.pipelines:cli_serve
  :prog: max serve
  :hide-description:
