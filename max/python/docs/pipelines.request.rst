:title: max.pipelines.request
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.pipelines.request
=====================

.. automodule:: max.pipelines.request
   :no-members:

.. currentmodule:: max.pipelines.request

Submodules
----------

.. toctree::
   :maxdepth: 1

   pipelines.request.provider_options

Base request types
------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Request
   RequestID
   RequestType

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/data.rst

   DUMMY_REQUEST_ID

Responses API request
---------------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   OpenResponsesRequest
   OpenResponsesRequestBody
   ResponseResource

Messages
--------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   AssistantMessage
   DeveloperMessage
   InputMessage
   Message
   MessageRole
   MessageStatus
   SystemMessage
   UserMessage

Input content
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   InputContent
   InputFileContent
   InputImageContent
   InputTextContent
   InputVideoContent
   ItemReferenceParam
   ImageDetail

Output content
--------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   OutputAudioContent
   OutputContent
   OutputImageContent
   OutputTextContent
   OutputTokensDetails
   OutputVideoContent
   RefusalContent

Tools
-----

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   AllowedToolChoice
   FunctionCall
   FunctionCallOutput
   FunctionCallStatus
   FunctionTool
   FunctionToolChoice
   FunctionToolParam
   ToolChoice
   ToolChoiceValueEnum

Response format
---------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   JsonObjectField
   JsonObjectParam
   JsonSchemaField
   JsonSchemaParam
   ResponseFormat
   ResponseFormatParam
   TextField
   TextParam

Reasoning
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ReasoningBody
   ReasoningEffortEnum
   ReasoningParam
   ReasoningReference
   ReasoningReferenceParam
   ReasoningSummaryContent
   ReasoningSummaryEnum

Usage and tokens
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   AudioGenerationDetails
   Error
   IncludeEnum
   IncompleteDetails
   InputTokensDetails
   LogProb
   ServiceTierEnum
   StreamOptionsParam
   TopLogProb
   TruncationEnum
   UrlCitationBody
   UrlCitationParam
   Usage
   VerbosityEnum

Protocols
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   FastAPIRequestProtocol

