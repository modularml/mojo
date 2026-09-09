:title: max.pipelines.context
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.pipelines.context
=====================

.. automodule:: max.pipelines.context
   :no-members:

.. currentmodule:: max.pipelines.context

Concrete context classes
------------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   TextContext
   TextAndVisionContext
   PixelContext
   AudioContext

Generation status
-----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   GenerationStatus

Constants
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/data.rst

   FUTURE_TOKEN

Context protocols
-----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   BaseContext

Type variables
--------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/attribute.rst

   BaseContextType
   TextGenerationContextType
   VLMContextType
   PixelGenerationContextType
   AudioGenerationContextType

Sampling
--------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   SamplingParams
   SamplingParamsInput
   SamplingParamsGenerationConfigDefaults

Output types
------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   TextGenerationOutput
   GenerationOutput
   TextGenerationResponseFormat
   LogProbabilities

Token management
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   TokenBuffer
   ImageMetadata
   Range
   TokenHashOverride

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/attribute.rst

   TokenSlice

Grammar and structured output
------------------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   GrammarEnforcementState
   GrammarEnforcementSnapshot
   GrammarMatcher
   StructuredOutputRegionDelimiters

Speculative decoding
--------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   SpecDecodingState

EOS tracking
------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   EOSTracker

Logits processors
-----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/attribute.rst

   LogitsProcessor
   BatchLogitsProcessor

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ProcessorInputs
   BatchProcessorInputs

Exceptions
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   InputError
   PromptTooLongError

Validation functions
--------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   validate_aspect_ratio_args
   validate_flux2_max_pixel_area
   validate_image_grid_thw_args
   validate_image_shape_5d
   validate_initial_prompt_has_image
   validate_only_one_image
   validate_requires_vision_context
   validate_vision_position_ids
   validate_wan_max_pixel_area
