:title: max.pipelines.weights
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.pipelines.weights
===============================

.. automodule:: max.pipelines.weights
   :no-members:

.. currentmodule:: max.pipelines.weights

HuggingFace integration
-----------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   HuggingFaceRepo

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   download_weight_files
   generate_local_model_path
   is_diffusion_pipeline
   try_to_load_from_cache
   validate_hf_repo_access

Weight loading
--------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   WeightPathParser

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   auto_cast_weights_from_env
   gptq_quant_config
   parse_quant_config
   resolve_hf_quant_config

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/data.rst

   AUTO_CAST_ENV_VAR
