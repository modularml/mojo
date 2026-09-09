:title: max.pipelines.diffusion
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.pipelines.diffusion
=======================

.. automodule:: max.pipelines.diffusion
   :no-members:

.. currentmodule:: max.pipelines.diffusion

Submodules
----------

.. toctree::
   :maxdepth: 1

   pipelines.diffusion.config
   pipelines.diffusion.schedulers

Pipelines
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   CompileWrapper
   DiffusionPipeline
   DiffusionPipelineOutput
   PixelGenerationPipeline

First-block cache
-----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   FirstBlockCache
   FirstBlockCacheState

Denoising cache
---------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   DenoisingCacheConfig
   DenoisingCacheSettings
   DenoisingCacheState
   TaylorSeer
   TaylorSeerBufferState
   TaylorSeerCache
   TaylorSeerState

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   fbcache_conditional_execution
   max_compile
   run_denoising_step

