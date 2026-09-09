:title: max list

Lists every pipeline architecture registered with MAX, along with example
Hugging Face repository IDs and the data-type encodings each architecture
supports. Use this command to discover what you can pass to ``max serve``,
``max generate``, or ``max encode``.

For example, list all available architectures:

.. code-block:: bash

    max list

The output is grouped by architecture, with the example repos and supported
encodings under each entry:

.. code-block:: text

    Architecture: Llama3
        Example Huggingface Repo Ids:
            modularai/Llama-3.1-8B-Instruct-GGUF
        Encoding Supported: float32
        Encoding Supported: bfloat16

To get the same information as JSON (for use in scripts or other tooling),
pass the ``--json`` flag:

.. code-block:: bash

    max list --json

The JSON output uses the structure
``{"architectures": {<name>: {"example_repo_ids": [...], "supported_encodings": [...]}}}``.

.. click:: max._entrypoints.pipelines:cli_list
  :prog: max list
  :hide-description:
