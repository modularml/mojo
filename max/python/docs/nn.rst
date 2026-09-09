:title: max.nn
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.nn
======

.. automodule:: max.nn
   :no-members:

.. currentmodule:: max.nn

Submodules
----------

.. toctree::
   :maxdepth: 1

   nn.attention
   nn.kernels
   nn.kv_cache

Base classes
------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Identity
   Layer
   LayerList
   Module
   Sequential
   Shardable
   Signals

Linear layers
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ColumnParallelLinear
   Embedding
   GPTQLinear
   Linear
   MLP
   StackedLinear
   VocabParallelEmbedding

Normalization
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ConstantLayerNorm
   GroupNorm
   LayerNorm
   RMSNorm

Rotary embeddings
-----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   DynamicRotaryEmbedding
   LinearScalingParams
   Llama3RopeScalingParams
   Llama3RotaryEmbedding
   LongRoPERotaryEmbedding
   LongRoPEScalingParams
   RotaryEmbedding
   YarnRotaryEmbedding
   YarnScalingParams

Transformer
-----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   DistributedTransformer
   DistributedTransformerBlock
   ReturnHiddenStates
   ReturnLogits
   Transformer
   TransformerBlock

Convolution
-----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Conv1D
   Conv2d
   Conv3D
   ConvTranspose1d
   WeightNormConvTranspose1d

Mixture of experts
------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Allreduce
   ~max.nn.comm.ep.EPCommBuffers
   MoE
   MoEGate
   MoEQuantized

Sampling
--------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   MinPSampler
   RejectionSampler
   RejectionSamplerWithResiduals

Quantization
------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   QuantConfig
   QuantFormat
   InputScaleSpec
   ScaleGranularity
   ScaleOrigin
   WeightScaleSpec

Hooks
-----

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ~max.nn.hooks.PrintHook

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   build_max_lengths_tensors
   clamp
   forward_moe_sharded_layers
   split_batch
   split_batch_replicated

