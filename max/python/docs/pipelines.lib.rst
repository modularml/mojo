:title: max.pipelines.lib
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.pipelines.lib
=================

.. automodule:: max.pipelines.lib
   :no-members:

.. currentmodule:: max.pipelines.lib

Submodules
----------

.. toctree::
   :maxdepth: 1

   pipelines.lib.interfaces
   pipelines.lib.log_probabilities

Configuration
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   DenoisingCacheConfig
   DenoisingCacheSettings
   KVConnectorConfig
   MAXConfig
   MAXModelConfigBase
   PipelineArgs
   PipelineRuntimeConfig

Pipelines
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   EmbeddingsPipelineType
   OverlapTextGenerationPipeline

Model interface
---------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   AlwaysSignalBuffersMixin
   PipelineModelWithKVCache
   UnifiedEagleOutputs

Tokenizers
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   AudioGenerationTokenizer
   PixelGenerationTokenizer

Utilities
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   CompilationTimer
   HuggingFaceRepo
   MemoryPlan
   ModelManifest
   RetrievedPipeline
   VisionPreprocessCache
   WeightPathParser

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   build_eos_tracker_for_request
   convert_max_config_value
   deep_merge_max_configs
   float32_array_to_buffer
   float32_to_bfloat16_as_uint16
   generate_local_model_path
   get_default_max_config_file_section_name
   max_tokens_to_generate
   parse_quant_config
   resolve_max_config_inheritance
   try_to_load_from_cache
   validate_hf_repo_access

