# PyTorch Layers to MAX Mapping Guide

- **Authors:** Brad Larson and Claude
- **Date:** June 23, 2025

## Introduction

This guide provides mappings between common PyTorch layers used in Hugging Face
`transformers` and their equivalent MAX graph operations and layer
abstractions.

## Table of Contents

1. [Overview](#overview)
2. [Core Layer Mappings](#core-layer-mappings)
3. [Graph Operations Mapping](#graph-operations-mapping)
4. [Implementation Examples](#implementation-examples)
5. [Performance Optimization Tips](#performance-optimization-tips)

## Overview

MAX provides two levels of abstraction for building neural networks:

- **High-level layers** (`max.nn`): PyTorch-compatible layer abstractions
- **Low-level graph operations** (`max.graph.ops`): Fine-grained tensor
  operations

### Key Differences from PyTorch

- MAX uses explicit device placement
- Supports advanced quantization (Float8, GPTQ)
- Provides distributed/sharded variants of common layers
- Offers hardware-optimized kernels for specific operations
- MAX relies on the construction, compilation, and execution of graphs, unlike
  PyTorch's eager execution

## Core Layer Mappings

### 1. Linear Layers

| HuggingFace/PyTorch    | MAX Layer                       | MAX Graph Op             | Notes                             |
|------------------------|---------------------------------|--------------------------|-----------------------------------|
| `nn.Linear`            | `max.nn.Linear`                 | `ops.matmul` + `ops.add` | MAX supports quantization options |
| `nn.Linear` (no bias)  | `max.nn.Linear(has_bias=False)` | `ops.matmul`             | Use `has_bias=False` parameter    |
| Column Parallel Linear | `max.nn.ColumnParallelLinear`   | -                        | For tensor parallelism            |
| GPTQ Quantized Linear  | `max.nn.GPTQLinear`             | -                        | GPTQ quantization support         |

**Example:**

```python
# PyTorch
linear = nn.Linear(768, 3072)

# MAX Layer
from max import nn

nn.linear = Linear(in_dim=768, out_dim=3072, dtype=DType.float32, device=device)

# MAX Graph Op
with Graph("linear") as g:
    x = ops.constant(...)
    weight = ops.constant(...)
    bias = ops.constant(...)
    output = ops.matmul(x, weight.T) + bias
```

### 2. Embedding Layers

| HuggingFace/PyTorch      | MAX Layer                       | MAX Graph Op | Notes                        |
|--------------------------|---------------------------------|--------------|------------------------------|
| `nn.Embedding`           | `max.nn.Embedding`              | `ops.gather` | Token embedding lookup       |
| Vocab Parallel Embedding | `max.nn.VocabParallelEmbedding` | -            | For distributed vocabularies |

**Example:**

```python
# PyTorch
embed = nn.Embedding(50000, 768)

# MAX Layer
from max import nn

embed = nn.Embedding(
    vocab_size=50000, hidden_dim=768, dtype=DType.float32, device=device
)
```

### 3. Normalization Layers

| HuggingFace/PyTorch | MAX Layer                   | MAX Graph Op          | Notes                       |
|---------------------|-----------------------------|-----------------------|-----------------------------|
| `nn.LayerNorm`      | `max.nn.LayerNorm`          | `ops.layer_norm`      | Epsilon parameter available |
| RMSNorm (custom)    | `max.nn.RMSNorm`            | Custom implementation | Used in Llama, Gemma        |
| `nn.GroupNorm`      | `max.nn.GroupNorm`          | Custom implementation | Group-wise normalization    |

**Example:**

```python
# PyTorch
norm = nn.LayerNorm(768, eps=1e-5)

# MAX Layer
from max import nn

norm = nn.LayerNorm(dims=768, eps=1e-5, device=device, dtype=DType.float32)

# MAX Graph Op
with Graph("layernorm") as g:
    x = ops.constant(...)
    gamma = ops.constant(...)
    beta = ops.constant(...)
    normalized = ops.layer_norm(x, gamma, beta, epsilon=1e-5)
```

### 4. Attention Mechanisms

| HuggingFace/PyTorch     | MAX Layer                                | MAX Graph Op | Notes                         |
|-------------------------|------------------------------------------|--------------|-------------------------------|
| `nn.MultiheadAttention` | `max.nn.MultiheadAttention`              | Multiple ops | Full attention implementation |
| Attention with RoPE     | `max.nn.AttentionWithRope`               | -            | Rotary position embeddings    |
| Distributed Attention   | `max.nn.TensorParallelAttentionWithRope` | -            | Multi-GPU attention           |
| Quantized Attention     | `max.nn.GPTQAttentionWithRope`           | -            | GPTQ quantized attention      |

**Attention Implementation with Graph Ops:**

```python
# Attention scores
scores = ops.matmul(Q, K.transpose(-2, -1)) / ops.sqrt(head_dim)

# Apply mask
if mask is not None:
    scores = ops.where(mask, scores, -float("inf"))

# Softmax
attention_weights = ops.softmax(scores)

# Apply attention to values
output = ops.matmul(attention_weights, V)
```

### 5. Activation Functions

| HuggingFace/PyTorch | MAX Layer | MAX Graph Op        | Notes                        |
|---------------------|-----------|---------------------|------------------------------|
| `F.gelu`            | -         | `ops.gelu`          | Supports approximation modes |
| `F.silu` / SwiGLU   | -         | `ops.silu`          | Sigmoid Linear Unit          |
| `F.sigmoid`         | -         | `ops.sigmoid`       | Sigmoid activation           |
| `F.tanh`            | -         | `ops.tanh`          | Hyperbolic tangent           |
| `F.relu`            | -         | `ops.maximum(x, 0)` | ReLU via maximum             |

**Example:**

```python
# PyTorch
output = F.gelu(input, approximate="tanh")

# MAX Graph Op
output = ops.gelu(input, approximate="tanh")
```

### 6. Positional Embeddings

| HuggingFace/PyTorch | MAX Layer                | MAX Graph Op         | Notes               |
|---------------------|--------------------------|----------------------|---------------------|
| Rotary Embeddings   | `max.nn.RotaryEmbedding` | Custom ops           | RoPE implementation |
| Sinusoidal PE       | -                        | `ops.sin`, `ops.cos` | Build with trig ops |
| Learnable PE        | `max.nn.Embedding`       | -                    | Use embedding layer |

### 7. Pooling and Reduction

| HuggingFace/PyTorch     | MAX Layer | MAX Graph Op | Notes                     |
|-------------------------|-----------|--------------|---------------------------|
| `F.adaptive_avg_pool1d` | -         | `ops.mean`   | Use with appropriate axis |
| `torch.mean`            | -         | `ops.mean`   | Reduction operation       |
| `torch.max`             | -         | `ops.max`    | Maximum reduction         |
| `torch.sum`             | -         | `ops.sum`    | Sum reduction             |

## Graph Operations Mapping

### Tensor Manipulation

| PyTorch Operation | MAX Graph Operation | Notes                       |
|-------------------|---------------------|-----------------------------|
| `torch.reshape`   | `ops.reshape`       | Shape inference with -1     |
| `torch.transpose` | `ops.transpose`     | Swap two dimensions         |
| `torch.permute`   | `ops.permute`       | Reorder all dimensions      |
| `torch.squeeze`   | `ops.squeeze`       | Remove dimensions of size 1 |
| `torch.unsqueeze` | `ops.unsqueeze`     | Add dimension of size 1     |
| `torch.cat`       | `ops.concat`        | Concatenate along axis      |
| `torch.stack`     | `ops.stack`         | Stack along new axis        |
| `torch.split`     | `ops.split`         | Split into chunks           |

### Mathematical Operations

| PyTorch Operation    | MAX Graph Operation | Notes                       |
|----------------------|---------------------|-----------------------------|
| `@` / `torch.matmul` | `ops.matmul`        | Matrix multiplication       |
| `+`                  | `ops.add`           | Element-wise addition       |
| `-`                  | `ops.sub`           | Element-wise subtraction    |
| `*`                  | `ops.mul`           | Element-wise multiplication |
| `/`                  | `ops.div`           | Element-wise division       |
| `torch.exp`          | `ops.exp`           | Exponential                 |
| `torch.log`          | `ops.log`           | Natural logarithm           |
| `torch.sqrt`         | `ops.sqrt`          | Square root                 |
| `torch.pow`          | `ops.pow`           | Power operation             |

### Indexing and Selection

| PyTorch Operation | MAX Graph Operation | Notes                    |
|-------------------|---------------------|--------------------------|
| `tensor[...]`     | `ops.slice_tensor`  | Advanced slicing         |
| `torch.gather`    | `ops.gather`        | Gather along dimension   |
| `torch.scatter`   | `ops.scatter`       | Scatter values           |
| `torch.where`     | `ops.where`         | Conditional selection    |
| `torch.topk`      | `ops.top_k`         | Top-k values and indices |

## Implementation Examples

### 1. Transformer Block in MAX

```python
from max import nn
from max.graph import ops


class TransformerBlockMAX:
    def __init__(self, config):
        # High-level MAX implementation
        self.attention = nn.AttentionWithRope(
            num_attention_heads=config.num_heads,
            hidden_size=config.hidden_size,
            device=config.device,
            dtype=config.dtype,
        )

        self.attention_norm = nn.RMSNorm(
            dim=config.hidden_size, dtype=config.dtype
        )

        self.mlp = nn.Sequential(
            [
                nn.Linear(config.hidden_size, config.intermediate_size),
                # Activation handled in forward
                nn.Linear(config.intermediate_size, config.hidden_size),
            ]
        )

        self.mlp_norm = nn.RMSNorm(dim=config.hidden_size, dtype=config.dtype)

    def forward(self, x, mask=None):
        # Attention block
        normed = self.attention_norm(x)
        attn_output = self.attention(normed, mask=mask)
        x = x + attn_output

        # MLP block
        normed = self.mlp_norm(x)
        mlp_output = self.mlp(normed)
        mlp_output = ops.gelu(mlp_output)  # Activation
        x = x + mlp_output

        return x
```

### 2. Multi-Head Attention with Graph Ops

```python
from max.graph import Graph, ops
from max.dtype import DType


def multi_head_attention_graph(
    query, key, value, num_heads, head_dim, mask=None
):
    """Multi-head attention using MAX graph operations."""
    batch_size, seq_len, hidden_dim = query.shape

    # Reshape for multi-head attention
    # [batch, seq, hidden] -> [batch, heads, seq, head_dim]
    Q = query.reshape((batch_size, seq_len, num_heads, head_dim))
    Q = Q.transpose(1, 2)

    K = key.reshape((batch_size, seq_len, num_heads, head_dim))
    K = K.transpose(1, 2)

    V = value.reshape((batch_size, seq_len, num_heads, head_dim))
    V = V.transpose(1, 2)

    # Attention scores
    scores = ops.matmul(Q, K.transpose(-2, -1))
    scores = scores / ops.sqrt(ops.constant(head_dim, dtype=DType.float32))

    # Apply mask if provided
    if mask is not None:
        mask_value = ops.constant(-1e9, dtype=scores.dtype)
        scores = ops.where(mask, scores, mask_value)

    # Softmax
    attention_weights = ops.softmax(scores)

    # Apply attention to values
    context = ops.matmul(attention_weights, V)

    # Reshape back
    context = context.transpose(1, 2)
    context = context.reshape((batch_size, seq_len, hidden_dim))

    return context
```

### 3. Feed-Forward Network with Quantization

```python
from max import nn
from max.graph import ops
from max.dtype import DType


class FeedForwardMAX:
    def __init__(
        self, hidden_size, intermediate_size, use_float8=False, device=None
    ):

        # Configure Float8 if requested
        quant_config = None
        if use_float8:
            quant_config = QuantConfig(
                input_scale=nn.InputScaleSpec(
                    granularity=nn.ScaleGranularity.COLWISE,
                    origin=nn.ScaleOrigin.DYNAMIC,
                    dtype=DType.float8_e4m3fn,
                ),
                weight_scale=nn.WeightScaleSpec(
                    granularity=nn.ScaleGranularity.ROWWISE,
                    dtype=DType.float32,
                ),
                mlp_quantized_layers=set(),
                attn_quantized_layers=set(),
            )

        self.w1 = nn.Linear(
            in_dim=hidden_size,
            out_dim=intermediate_size,
            dtype=DType.float32,
            device=device,
            quant_config=quant_config,
        )

        self.w2 = nn.Linear(
            in_dim=intermediate_size,
            out_dim=hidden_size,
            dtype=DType.float32,
            device=device,
            quant_config=quant_config,
        )

    def forward(self, x):
        # SwiGLU activation
        hidden = self.w1(x)
        hidden = ops.silu(hidden)
        output = self.w2(hidden)
        return output
```

## Performance Optimization Tips

### 1. Use Hardware-Specific Optimizations

```python
# Enable tensor core operations for NVIDIA GPUs
from max.driver import Device

device = Device.gpu()
dtype = DType.float16  # Use half precision for tensor cores
```

### 2. Leverage Quantization

```python
# Use Float8 quantization for memory efficiency
from max import nn
from max.dtype import DType

config = QuantConfig(
    input_scale=nn.InputScaleSpec(
        granularity=nn.ScaleGranularity.BLOCK,
        origin=nn.ScaleOrigin.DYNAMIC,
        dtype=DType.float32,
        block_size=(1, 128),
    ),
    weight_scale=nn.WeightScaleSpec(
        granularity=nn.ScaleGranularity.BLOCK,
        dtype=DType.float32,
        block_size=(128, 128),
    ),
    mlp_quantized_layers=set(),
    attn_quantized_layers=set(),
)

linear = nn.Linear(..., quant_config=config)
```

### 3. Use Fused Operations

```python
# Instead of separate ops, use fused kernels where available
# Bad: separate normalization steps
mean = ops.mean(x, axis=-1, keepdims=True)
var = ops.mean(ops.pow(x - mean, 2), axis=-1, keepdims=True)
normalized = (x - mean) / ops.sqrt(var + eps)

# Good: use built-in layer norm
normalized = ops.layer_norm(x, gamma, beta, epsilon=eps)
```

### 4. Optimize Memory Layout

```python
# Use contiguous memory layouts
# Reshape operations that don't require data movement
x_reshaped = x.reshape(new_shape)  # Efficient
x_transposed = x.transpose(0, 2, 1)  # May require copy
```

### 5. Batch Operations

```python
# Batch multiple operations together
with Graph("transformer") as g:
    # Define all operations in one graph
    # MAX will optimize execution order
    pass
```

### 6. Use Distributed Variants for Large Models

```python
# For multi-GPU setups
from max import nn

transformer = nn.DistributedTransformer(
    devices=[Device.gpu(0), Device.gpu(1)],
    ...
)
```

## Common Patterns and Best Practices

### 1. Residual Connections

```python
# Pattern: x + sublayer(x)
residual = x
x = layer_norm(x)
x = attention(x)
x = x + residual  # Use ops.add for graph ops
```

### 2. Attention Masking

```python
# Create causal mask
mask = ops.band_part(
    ops.ones((seq_len, seq_len)),
    num_lower=-1,  # Keep all lower triangle
    num_upper=0,  # Remove upper triangle
)
```

### 3. Positional Encoding Integration

```python
# Add positional embeddings
token_embeds = embedding(input_ids)
pos_embeds = positional_encoding(positions)
embeddings = ops.add(token_embeds, pos_embeds)
```

## References

For the latest updates and additional operations, refer to:

- MAX Python API docs: <https://max.modular.com/api/python>
- MAX Graph Operations: <https://max.modular.com/graph/ops/>
- MAX Neural Network Layers: <https://max.modular.com/api/python/nn>
