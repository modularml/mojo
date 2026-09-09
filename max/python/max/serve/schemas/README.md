# MAX Serve Schemas

Pydantic models that describe MAX Serve's HTTP request/response payloads.

## OpenAI

`openai.py` defines the schemas used by the `/v1/...` OpenAI-compatible
routes. Response models come straight from the
[`openai`](https://pypi.org/project/openai/) Python SDK so MAX Serve outputs
match the OpenAI wire format. Request models are defined locally because the
SDK only ships request shapes as `TypedDict` "params" types, which are
inconvenient to use as FastAPI request bodies. The local request models
mirror those shapes and add MAX-specific extensions (e.g. `target_endpoint`,
`dkv_cache_hint`, `top_k`, `min_tokens`).

To accept new OpenAI fields, just bump the pinned `openai` SDK version: the
new attributes flow through automatically. To add a MAX extension to a
request, edit the relevant Pydantic class in `openai.py` directly.
