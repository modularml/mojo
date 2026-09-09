:title: max encode

Converts input text into embeddings for semantic search, text similarity, and
NLP applications.

For example, encode a sentence with a Sentence-Transformers model:

.. code-block:: bash

    max encode \
      --model sentence-transformers/all-MiniLM-L6-v2 \
      --prompt "Convert this text into embeddings"

The command prints the embedding vector and timing for the run. Pair it with
``max list`` to see which encoder architectures MAX supports.

.. click:: max._entrypoints.pipelines:encode
  :prog: max encode
  :hide-description: