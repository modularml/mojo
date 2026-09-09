:title: max.pipelines.architectures
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.pipelines.architectures
============================

.. automodule:: max.pipelines.architectures
   :no-members:

MAX includes built-in support for a wide range of model architectures. Each
architecture module registers a
:class:`~max.pipelines.lib.registry.SupportedArchitecture` instance that tells
the pipeline system how to load, configure, and execute a particular model
family.

.. toctree::
   :hidden:

   pipelines.architectures.bert
   pipelines.architectures.deepseekV2
   pipelines.architectures.deepseekV3
   pipelines.architectures.deepseekV3_2
   pipelines.architectures.deepseekV3_nextn
   pipelines.architectures.dflash_llama3
   pipelines.architectures.diffusion_gemma
   pipelines.architectures.dspark_draft
   pipelines.architectures.eagle3_deepseekV3
   pipelines.architectures.eagle_llama3
   pipelines.architectures.flux2
   pipelines.architectures.gemma3
   pipelines.architectures.gemma3multimodal
   pipelines.architectures.gemma4
   pipelines.architectures.gemma4_assistant
   pipelines.architectures.glm5_1
   pipelines.architectures.gpt_oss
   pipelines.architectures.granite
   pipelines.architectures.hy_v3
   pipelines.architectures.idefics3
   pipelines.architectures.ideogram4
   pipelines.architectures.inkling
   pipelines.architectures.internvl
   pipelines.architectures.kimik2_5
   pipelines.architectures.laguna
   pipelines.architectures.lfm2
   pipelines.architectures.llama3
   pipelines.architectures.llama4
   pipelines.architectures.mamba
   pipelines.architectures.minimax_m2
   pipelines.architectures.minimax_music3
   pipelines.architectures.mistral
   pipelines.architectures.mistral3
   pipelines.architectures.mpnet
   pipelines.architectures.nemotron_h
   pipelines.architectures.olmo
   pipelines.architectures.olmo2
   pipelines.architectures.olmo3
   pipelines.architectures.phi3
   pipelines.architectures.pixtral
   pipelines.architectures.qwen2
   pipelines.architectures.qwen2_5vl
   pipelines.architectures.qwen3
   pipelines.architectures.qwen3_5
   pipelines.architectures.qwen3_embedding
   pipelines.architectures.qwen3vl_moe
   pipelines.architectures.qwen_image
   pipelines.architectures.qwen_image_edit
   pipelines.architectures.step3p5
   pipelines.architectures.unified_dflash_gemma4_31b
   pipelines.architectures.unified_dflash_kimi_k25
   pipelines.architectures.unified_dflash_llama3
   pipelines.architectures.unified_dspark_gemma4_12b
   pipelines.architectures.unified_dspark_gemma4_31b
   pipelines.architectures.unified_eagle_llama3
   pipelines.architectures.unified_mtp_deepseekV3
   pipelines.architectures.unified_mtp_gemma4
   pipelines.architectures.unified_mtp_glm5_2
   pipelines.architectures.unified_mtp_inkling
   pipelines.architectures.unified_mtp_qwen3_5
   pipelines.architectures.wan

Text generation
---------------

.. autosummary::
   :nosignatures:

   ~max.pipelines.architectures.deepseekV2
   ~max.pipelines.architectures.deepseekV3
   ~max.pipelines.architectures.deepseekV3_2
   ~max.pipelines.architectures.deepseekV3_nextn
   ~max.pipelines.architectures.dflash_llama3
   ~max.pipelines.architectures.diffusion_gemma
   ~max.pipelines.architectures.dspark_draft
   ~max.pipelines.architectures.eagle3_deepseekV3
   ~max.pipelines.architectures.eagle_llama3
   ~max.pipelines.architectures.gemma3
   ~max.pipelines.architectures.gemma3multimodal
   ~max.pipelines.architectures.gemma4
   ~max.pipelines.architectures.gemma4_assistant
   ~max.pipelines.architectures.glm5_1
   ~max.pipelines.architectures.gpt_oss
   ~max.pipelines.architectures.granite
   ~max.pipelines.architectures.hy_v3
   ~max.pipelines.architectures.idefics3
   ~max.pipelines.architectures.inkling
   ~max.pipelines.architectures.internvl
   ~max.pipelines.architectures.kimik2_5
   ~max.pipelines.architectures.laguna
   ~max.pipelines.architectures.lfm2
   ~max.pipelines.architectures.llama3
   ~max.pipelines.architectures.llama4
   ~max.pipelines.architectures.mamba
   ~max.pipelines.architectures.minimax_m2
   ~max.pipelines.architectures.mistral
   ~max.pipelines.architectures.mistral3
   ~max.pipelines.architectures.nemotron_h
   ~max.pipelines.architectures.olmo
   ~max.pipelines.architectures.olmo2
   ~max.pipelines.architectures.olmo3
   ~max.pipelines.architectures.phi3
   ~max.pipelines.architectures.pixtral
   ~max.pipelines.architectures.qwen2
   ~max.pipelines.architectures.qwen2_5vl
   ~max.pipelines.architectures.qwen3
   ~max.pipelines.architectures.qwen3_5
   ~max.pipelines.architectures.qwen3vl_moe
   ~max.pipelines.architectures.step3p5
   ~max.pipelines.architectures.unified_dflash_gemma4_31b
   ~max.pipelines.architectures.unified_dflash_kimi_k25
   ~max.pipelines.architectures.unified_dflash_llama3
   ~max.pipelines.architectures.unified_dspark_gemma4_12b
   ~max.pipelines.architectures.unified_dspark_gemma4_31b
   ~max.pipelines.architectures.unified_eagle_llama3
   ~max.pipelines.architectures.unified_mtp_deepseekV3
   ~max.pipelines.architectures.unified_mtp_gemma4
   ~max.pipelines.architectures.unified_mtp_glm5_2
   ~max.pipelines.architectures.unified_mtp_inkling
   ~max.pipelines.architectures.unified_mtp_qwen3_5

Embeddings
----------

.. autosummary::
   :nosignatures:

   ~max.pipelines.architectures.bert
   ~max.pipelines.architectures.mpnet
   ~max.pipelines.architectures.qwen3_embedding

Image generation
----------------

.. autosummary::
   :nosignatures:

   ~max.pipelines.architectures.flux2
   ~max.pipelines.architectures.ideogram4
   ~max.pipelines.architectures.qwen_image
   ~max.pipelines.architectures.qwen_image_edit
   ~max.pipelines.architectures.wan

Audio generation
----------------

.. autosummary::
   :nosignatures:

   ~max.pipelines.architectures.minimax_music3
