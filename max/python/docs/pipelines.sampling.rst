:title: max.pipelines.sampling
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.pipelines.sampling
======================

.. automodule:: max.pipelines.sampling
   :no-members:

.. currentmodule:: max.pipelines.sampling

Configuration
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   SamplingConfig

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/data.rst

   DEFAULT_STRUCTURED_OUTPUT_BACKEND

Processors
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   FrequencyData
   FusedSamplingProcessor
   PenaltyInputs
   SamplerInputs

Samplers
--------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   RejectionRunner
   SyntheticRunner
   TokenSampler

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   rejection_runner_registry
   rejection_sampler
   rejection_sampler_with_residuals
   token_sampler

Logits processing
-----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   apply_logits_processors
   build_greedy_acceptance_sampler_graph
   build_stochastic_acceptance_sampler_graph
   build_synthetic_acceptance_sampler_graph
