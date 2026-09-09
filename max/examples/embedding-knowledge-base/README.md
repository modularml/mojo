# Knowledge base with a text embedding model

A document management system that uses embeddings and clustering to organize and
search through documents semantically.

1. Start the embeddings endpoint:

    ```bash
    max serve --model sentence-transformers/all-mpnet-base-v2
    ```

1. Run the system:

    ```bash
    python -m embeddings.kb_system
    ```

For a complete walkthrough, see the tutorial to [Create a knowledge base with a
text embedding
model](https://max.modular.com/develop/run-embeddings-with-max-serve)
