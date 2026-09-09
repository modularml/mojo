"""Resource estimates for tests, generated internally by utils/generate_test_resources_report"""
TEST_RESOURCES = {
    "//max/examples/capi:test": {
        "default": {
            "cpu": 2,
            "memory": 1464,
        },
    },
    "//max/examples/custom_ops:addition.example-test": {
        "default": {
            "cpu": 2,
            "memory": 1409,
        },
    },
    "//max/examples/custom_ops:histogram.example-test": {
        "default": {
            "cpu": 2,
            "memory": 1445,
        },
    },
    "//max/examples/custom_ops:image_pipeline.example-test": {
        "default": {
            "cpu": 2,
            "memory": 1494,
        },
    },
    "//max/examples/custom_ops:mandelbrot.example-test": {
        "default": {
            "cpu": 2,
            "memory": 1422,
        },
    },
    "//max/examples/custom_ops:parametric_addition.example-test": {
        "default": {
            "cpu": 2,
            "memory": 1402,
        },
    },
    "//max/examples/custom_ops:top_k.example-test": {
        "default": {
            "cpu": 2,
            "memory": 1939,
        },
    },
    "//max/examples/custom_ops:vector_addition.example-test": {
        "default": {
            "cpu": 2,
            "memory": 1261,
        },
    },
    "//max/examples/pytorch_custom_ops:addition.example-test": {
        "default": {
            "cpu": 2,
            "memory": 1822,
        },
    },
    "//max/examples/pytorch_custom_ops:graph.example-test": {
        "default": {
            "memory": 3686,
        },
    },
    "//max/examples/pytorch_custom_ops:grayscale.example-test": {
        "default": {
            "cpu": 2,
            "memory": 1888,
        },
    },
    "//max/kernels/benchmarks/autotune:tests/test_kbench": {
        "default": {
            "cpu": 3,
            "memory": 2262,
        },
    },
    "//max/kernels/benchmarks:algorithm/parallelize_overhead.mojo.test": {
        "default": {
            "memory": 960,
        },
    },
    "//max/kernels/test/gpu/linalg:test_matmul_sm100_ptx.mojo.test": {
        "default": {
            "memory": 360,
        },
    },
    "//max/kernels/test/kv_cache:test_mha_mixed_ce_tg.mojo.test": {
        "default": {
            "memory": 232,
        },
    },
    "//max/kernels/test/linalg:test_gemv.mojo.test": {
        "default": {
            "cpu": 8,
            "memory": 233,
        },
    },
    "//max/kernels/test/linalg:test_neon_dotprod_intrinsics.mojo.test": {
        "default": {
            "memory": 360,
        },
    },
    "//max/kernels/test/linalg:test_neon_matmul_intrinsics.mojo.test": {
        "default": {
            "memory": 360,
        },
    },
    "//max/kernels/test/linalg:test_vnni_intrinsics.mojo.test": {
        "default": {
            "memory": 488,
        },
    },
    "//max/kernels/test/nn:test_conv1d.mojo.test": {
        "default": {
            "memory": 130,
        },
    },
    "//max/kernels/test/nn:test_direct_conv.mojo.test": {
        "default": {
            "memory": 808,
        },
    },
    "//max/kernels/test/nn:test_toppminp.mojo.test": {
        "default": {
            "memory": 209,
        },
    },
    "//max/kernels/test/quantization:test_qmatmul_k.mojo.test": {
        "default": {
            "cpu": 4,
        },
    },
    "//max/tests/integration/graph:test_matmul_packed": {
        "default": {
            "memory": 5897,
        },
    },
    "//max/tests/integration/graph:test_reduce_add": {
        "default": {
            "memory": 1890,
        },
    },
    "//max/tests/integration/interfaces:test_hash_image": {
        "default": {
            "cpu": 6,
            "memory": 360,
        },
    },
    "//max/tests/integration/interfaces:test_queue": {
        "default": {
            "cpu": 5,
            "memory": 360,
        },
    },
    "//max/tests/integration/interfaces:test_serialization": {
        "default": {
            "cpu": 5,
            "memory": 360,
        },
    },
    "//max/tests/integration/interfaces:test_tokens": {
        "default": {
            "cpu": 5,
            "memory": 360,
        },
    },
    "//max/tests/integration/interfaces:text_generation/test_text_generation_request": {
        "default": {
            "cpu": 5,
            "memory": 360,
        },
    },
    "//max/tests/integration/nn/module_v3:norm/test_rms_norm": {
        "default": {
            "cpu": 3,
            "memory": 21602,
        },
    },
    "//max/tests/integration/nn/module_v3:rope/test_rope": {
        "default": {
            "cpu": 4,
            "memory": 29830,
        },
    },
    "//max/tests/integration/nn/module_v3:test_embedding": {
        "default": {
            "cpu": 4,
            "memory": 16103,
        },
    },
    "//max/tests/integration/nn/module_v3:test_linear": {
        "default": {
            "cpu": 3,
            "memory": 15871,
        },
    },
    "//max/tests/integration/nn/module_v3:test_module": {
        "default": {
            "cpu": 7,
            "memory": 42461,
        },
    },
    "//max/tests/integration/nn/module_v3:test_sequential": {
        "default": {
            "cpu": 3,
            "memory": 5680,
        },
    },
    "//max/tests/integration/tensor:test_arange": {
        "default": {
            "cpu": 4,
            "memory": 43294,
        },
    },
    "//max/tests/integration/tensor:test_functional_binary": {
        "default": {
            "cpu": 5,
            "memory": 44612,
        },
    },
    "//max/tests/integration/tensor:test_functional_custom": {
        "default": {
            "cpu": 8,
            "memory": 4387,
        },
    },
    "//max/tests/integration/tensor:test_functional_other": {
        "default": {
            "cpu": 7,
            "memory": 117646,
        },
    },
    "//max/tests/integration/tensor:test_functional_reduction": {
        "default": {
            "cpu": 8,
            "memory": 13848,
        },
    },
    "//max/tests/integration/tensor:test_functional_unary": {
        "default": {
            "cpu": 6,
            "memory": 58646,
        },
    },
    "//max/tests/integration/tensor:test_random": {
        "default": {
            "cpu": 5,
            "memory": 29461,
        },
    },
    "//max/tests/integration/tensor:test_tensor_elemwise": {
        "default": {
            "cpu": 7,
            "memory": 111186,
        },
    },
    "//max/tests/integration/tensor:test_tensor_matmul": {
        "default": {
            "cpu": 5,
            "memory": 12228,
        },
    },
    "//max/tests/integration/tensor:test_tensor_repr": {
        "default": {
            "cpu": 7,
            "memory": 42664,
        },
    },
    "//max/tests/integration/unorganized:test_load_library": {
        "default": {
            "cpu": 5,
        },
    },
    "//max/tests/integration/unorganized:test_load_library_3.10": {
        "default": {
            "cpu": 7,
        },
    },
    "//max/tests/integration/unorganized:test_load_library_3.11": {
        "default": {
            "cpu": 7,
        },
    },
    "//max/tests/integration/unorganized:test_load_library_3.13": {
        "default": {
            "cpu": 7,
            "memory": 127,
        },
    },
    "//max/tests/integration/unorganized:test_load_library_3.14": {
        "default": {
            "cpu": 6,
        },
    },
    "//max/tests/integration/unorganized:tests-fail-weight-loading": {
        "default": {
            "cpu": 5,
        },
    },
    "//max/tests/integration/dataprocessing:test_causal_attention_mask": {
        "default": {
            "memory": 360,
        },
    },
    "//max/tests/integration/dataprocessing:test_collate_batch": {
        "default": {
            "cpu": 2,
            "memory": 360,
        },
    },
    "//max/tests/integration/dataprocessing:test_max_tokens_to_generate": {
        "default": {
            "cpu": 5,
            "memory": 360,
        },
    },
    "//max/tests/integration/kv_cache/attention:attention_no_opaque_tests": {
        "default": {
            "cpu": 2,
            "memory": 4325,
        },
    },
    "//max/tests/integration/kv_cache/attention:attention_tests": {
        "default": {
            "cpu": 2,
            "memory": 6932,
        },
    },
    "//max/tests/integration/kv_cache/transfer_engine:test_notification_latency": {
        "default": {
            "cpu": 2,
            "memory": 2078,
        },
    },
    "//max/tests/integration/kv_cache/transfer_engine:test_send_recv": {
        "default": {
            "cpu": 3,
            "memory": 710,
        },
    },
    "//max/tests/integration/kv_cache:embedding": {
        "default": {
            "cpu": 8,
            "memory": 21692,
        },
    },
    "//max/tests/integration/kv_cache:test_kv_cache_matmul": {
        "default": {
            "cpu": 2,
            "memory": 19446,
        },
    },
    "//max/tests/integration/kv_cache:test_memory_estimation": {
        "default": {
            "cpu": 2,
            "memory": 295,
        },
    },
    "//max/tests/integration/kv_cache:test_prefix_caching": {
        "default": {
            "memory": 2448,
        },
    },
    "//max/tests/integration/kv_cache:test_print_kv_cache": {
        "default": {
            "cpu": 2,
            "memory": 13339,
        },
    },
    "//max/tests/integration/kv_cache:test_rms_norm_key_cache": {
        "default": {
            "cpu": 2,
            "memory": 5705,
        },
    },
    "//max/tests/integration/architectures/mistral3:tests": {
        "default": {
            "cpu": 2,
            "memory": 721,
        },
    },
    "//max/tests/integration/nn/kv_cache:test_block_hasher": {
        "default": {
            "cpu": 2,
            "memory": 875,
        },
    },
    "//max/tests/integration/nn/kv_cache:test_cache_params": {
        "default": {
            "cpu": 2,
            "memory": 296,
        },
    },
    "//max/tests/integration/nn/kv_cache:test_data_parallelism_utils": {
        "default": {
            "memory": 295,
        },
    },
    "//max/tests/integration/nn/kv_cache:test_kv_cache_manager": {
        "default": {
            "memory": 2587,
        },
    },
    "//max/tests/integration/nn/norm:norm_tests": {
        "default": {
            "memory": 3623,
        },
    },
    "//max/tests/integration/nn:test_conv": {
        "default": {
            "cpu": 3,
            "memory": 21245,
        },
    },
    "//max/tests/integration/nn:test_identity": {
        "default": {
            "cpu": 8,
            "memory": 8041,
        },
    },
    "//max/tests/integration/nn:test_layer_hook": {
        "default": {
            "cpu": 8,
            "memory": 8842,
        },
    },
    "//max/tests/integration/nn:test_mlp": {
        "default": {
            "cpu": 8,
            "memory": 55325,
        },
    },
    "//max/tests/integration/nn:test_print_hook": {
        "default": {
            "cpu": 6,
            "memory": 10141,
        },
    },
    "//max/tests/integration/pipelines:test_compute_log_probabilities": {
        "default": {
            "cpu": 2,
            "memory": 2149,
        },
    },
    "//max/tests/integration/pipelines:test_lora_graph_inputs": {
        "default": {
            "cpu": 2,
            "memory": 859,
        },
    },
    "//max/tests/integration/pipelines:test_pipeline_lora_sorting": {
        "default": {
            "cpu": 2,
            "memory": 896,
        },
    },
    "//max/tests/integration/pipelines:test_text_generation_pipeline": {
        "default": {
            "cpu": 2,
            "memory": 2246,
        },
    },
    "//max/tests/integration/architectures/qwen2_5vl:test_compute_scatter_gather_indices": {
        "default": {
            "cpu": 2,
            "memory": 776,
        },
    },
    "//max/tests/integration/architectures/qwen2_5vl:test_vision_functions": {
        "default": {
            "memory": 14933,
        },
    },
    "//max/tests/integration/architectures/qwen3vl:test_vision_functions": {
        "default": {
            "cpu": 2,
            "memory": 935,
        },
    },
    "//max/tests/integration/architectures/whisper:whisper": {
        "default": {
            "cpu": 2,
            "memory": 935,
        },
    },
    "//max/tests/integration/accuracy:test_compare_tensors": {
        "default": {
            "memory": 2370,
        },
    },
    "//max/tests/integration/accuracy:test_debug_model": {
        "default": {
            "cpu": 2,
            "memory": 893,
        },
    },
    "//max/tests/integration/accuracy:test_debug_utils": {
        "default": {
            "cpu": 2,
            "memory": 911,
        },
    },
    "//max/tests/integration/tools:test_hf_config_overrides": {
        "default": {
            "cpu": 2,
            "memory": 743,
        },
    },
    "//max/tests/integration/cli:test_pipelines_cli_help": {
        "default": {
            "cpu": 2,
            "memory": 775,
        },
    },
    "//max/tests/integration/cli:test_pipelines_cli_json_lightweight": {
        "default": {
            "memory": 748,
        },
    },
    "//max/tests/integration/cli:test_pipelines_cli_lightweight": {
        "default": {
            "cpu": 2,
            "memory": 778,
        },
    },
    "//max/tests/integration/accuracy:test_pipelines_lm_eval": {
        "default": {
            "cpu": 2,
            "memory": 8350,
        },
    },
    "//max/tests/integration/serve:test_sagemaker_cpu": {
        "default": {
            "cpu": 2,
            "memory": 1256,
        },
    },
    "//max/tests/integration/serve:test_stop_cpu": {
        "default": {
            "cpu": 2,
            "memory": 1268,
        },
    },
    "//max/tests/tests/_core_mojo:tests": {
        "default": {
            "cpu": 3,
            "memory": 625,
        },
    },
    "//max/tests/tests/driver:test_device": {
        "default": {
            "cpu": 2,
            "memory": 262,
        },
    },
    "//max/tests/tests/driver:test_driver": {
        "default": {
            "cpu": 2,
            "memory": 263,
        },
    },
    "//max/tests/tests/driver:test_tensor": {
        "default": {
            "cpu": 2,
            "memory": 266,
        },
    },
    "//max/tests/tests/entrypoints:tests": {
        "default": {
            "cpu": 2,
            "memory": 754,
        },
    },
    "//max/tests/tests/graph:multi_version_tests": {
        "default": {
            "memory": 360,
        },
    },
    "//max/tests/tests/graph:multi_version_tests_3.10": {
        "default": {
            "cpu": 2,
            "memory": 344,
        },
    },
    "//max/tests/tests/graph:multi_version_tests_3.11": {
        "default": {
            "cpu": 3,
            "memory": 331,
        },
    },
    "//max/tests/tests/graph:multi_version_tests_3.13": {
        "default": {
            "cpu": 3,
            "memory": 341,
        },
    },
    "//max/tests/tests/graph:multi_version_tests_3.14": {
        "default": {
            "cpu": 2,
            "memory": 391,
        },
    },
    "//max/tests/tests/graph:ops/elementwise/test_atanh": {
        "default": {
            "cpu": 2,
            "memory": 293,
        },
    },
    "//max/tests/tests/graph:ops/elementwise/test_div": {
        "default": {
            "memory": 4972,
        },
    },
    "//max/tests/tests/graph:ops/elementwise/test_gelu": {
        "default": {
            "cpu": 2,
            "memory": 341,
        },
    },
    "//max/tests/tests/graph:ops/elementwise/test_is_inf": {
        "default": {
            "cpu": 2,
            "memory": 292,
        },
    },
    "//max/tests/tests/graph:ops/elementwise/test_is_nan": {
        "default": {
            "cpu": 2,
            "memory": 296,
        },
    },
    "//max/tests/tests/graph:ops/elementwise/test_logical_binary_ops": {
        "default": {
            "cpu": 2,
            "memory": 306,
        },
    },
    "//max/tests/tests/graph:ops/elementwise/test_logical_not": {
        "default": {
            "cpu": 2,
            "memory": 295,
        },
    },
    "//max/tests/tests/graph:ops/elementwise/test_sub": {
        "default": {
            "cpu": 2,
            "memory": 346,
        },
    },
    "//max/tests/tests/graph:ops/reduction/test_argminmax": {
        "default": {
            "cpu": 2,
            "memory": 295,
        },
    },
    "//max/tests/tests/graph:ops/test_allgather": {
        "default": {
            "cpu": 2,
            "memory": 370,
        },
    },
    "//max/tests/tests/graph:ops/test_allreduce": {
        "default": {
            "cpu": 2,
            "memory": 301,
        },
    },
    "//max/tests/tests/graph:ops/test_argsort": {
        "default": {
            "cpu": 2,
            "memory": 554,
        },
    },
    "//max/tests/tests/graph:ops/test_band_part": {
        "default": {
            "cpu": 2,
            "memory": 301,
        },
    },
    "//max/tests/tests/graph:ops/test_broadcast_to": {
        "default": {
            "cpu": 2,
            "memory": 317,
        },
    },
    "//max/tests/tests/graph:ops/test_buffer": {
        "default": {
            "memory": 1676,
        },
    },
    "//max/tests/tests/graph:ops/test_call": {
        "default": {
            "cpu": 2,
            "memory": 412,
        },
    },
    "//max/tests/tests/graph:ops/test_cast": {
        "default": {
            "cpu": 2,
            "memory": 295,
        },
    },
    "//max/tests/tests/graph:ops/test_chunk": {
        "default": {
            "cpu": 2,
            "memory": 383,
        },
    },
    "//max/tests/tests/graph:ops/test_complex": {
        "default": {
            "memory": 302,
        },
    },
    "//max/tests/tests/graph:ops/test_concat": {
        "default": {
            "memory": 308,
        },
    },
    "//max/tests/tests/graph:ops/test_conditional": {
        "default": {
            "memory": 319,
        },
    },
    "//max/tests/tests/graph:ops/test_constant": {
        "default": {
            "memory": 511,
        },
    },
    "//max/tests/tests/graph:ops/test_conv": {
        "default": {
            "memory": 308,
        },
    },
    "//max/tests/tests/graph:ops/test_conv3d": {
        "default": {
            "cpu": 2,
            "memory": 325,
        },
    },
    "//max/tests/tests/graph:ops/test_conv_transpose": {
        "default": {
            "memory": 347,
        },
    },
    "//max/tests/tests/graph:ops/test_cumsum": {
        "default": {
            "cpu": 2,
            "memory": 325,
        },
    },
    "//max/tests/tests/graph:ops/test_custom": {
        "default": {
            "memory": 997,
        },
    },
    "//max/tests/tests/graph:ops/test_device_chains_collectives": {
        "default": {
            "cpu": 2,
            "memory": 289,
        },
    },
    "//max/tests/tests/graph:ops/test_flatten": {
        "default": {
            "memory": 317,
        },
    },
    "//max/tests/tests/graph:ops/test_fold": {
        "default": {
            "cpu": 2,
            "memory": 574,
        },
    },
    "//max/tests/tests/graph:ops/test_gather": {
        "default": {
            "memory": 338,
        },
    },
    "//max/tests/tests/graph:ops/test_hann_window": {
        "default": {
            "memory": 292,
        },
    },
    "//max/tests/tests/graph:ops/test_irfft": {
        "default": {
            "cpu": 2,
            "memory": 828,
        },
    },
    "//max/tests/tests/graph:ops/test_layer_norm": {
        "default": {
            "memory": 300,
        },
    },
    "//max/tests/tests/graph:ops/test_linalg": {
        "default": {
            "cpu": 2,
            "memory": 454,
        },
    },
    "//max/tests/tests/graph:ops/test_min_max_overloads": {
        "default": {
            "cpu": 2,
            "memory": 302,
        },
    },
    "//max/tests/tests/graph:ops/test_nonzero": {
        "default": {
            "cpu": 2,
            "memory": 325,
        },
    },
    "//max/tests/tests/graph:ops/test_outer": {
        "default": {
            "cpu": 2,
            "memory": 354,
        },
    },
    "//max/tests/tests/graph:ops/test_pad": {
        "default": {
            "memory": 295,
        },
    },
    "//max/tests/tests/graph:ops/test_permute": {
        "default": {
            "cpu": 2,
            "memory": 305,
        },
    },
    "//max/tests/tests/graph:ops/test_quantized": {
        "default": {
            "cpu": 2,
            "memory": 557,
        },
    },
    "//max/tests/tests/graph:ops/test_random": {
        "default": {
            "cpu": 2,
            "memory": 386,
        },
    },
    "//max/tests/tests/graph:ops/test_range": {
        "default": {
            "cpu": 2,
            "memory": 438,
        },
    },
    "//max/tests/tests/graph:ops/test_rebind": {
        "default": {
            "memory": 308,
        },
    },
    "//max/tests/tests/graph:ops/test_reduction": {
        "default": {
            "cpu": 2,
            "memory": 342,
        },
    },
    "//max/tests/tests/graph:ops/test_repeat_interleave": {
        "default": {
            "memory": 581,
        },
    },
    "//max/tests/tests/graph:ops/test_reshape": {
        "default": {
            "memory": 349,
        },
    },
    "//max/tests/tests/graph:ops/test_resize": {
        "default": {
            "memory": 293,
        },
    },
    "//max/tests/tests/graph:ops/test_scatter": {
        "default": {
            "cpu": 2,
            "memory": 342,
        },
    },
    "//max/tests/tests/graph:ops/test_shape_to_tensor": {
        "default": {
            "cpu": 2,
            "memory": 313,
        },
    },
    "//max/tests/tests/graph:ops/test_slice": {
        "default": {
            "cpu": 2,
            "memory": 394,
        },
    },
    "//max/tests/tests/graph:ops/test_split": {
        "default": {
            "memory": 307,
        },
    },
    "//max/tests/tests/graph:ops/test_stack": {
        "default": {
            "cpu": 2,
            "memory": 377,
        },
    },
    "//max/tests/tests/graph:ops/test_tile": {
        "default": {
            "cpu": 2,
            "memory": 320,
        },
    },
    "//max/tests/tests/graph:ops/test_top_k": {
        "default": {
            "cpu": 2,
            "memory": 289,
        },
    },
    "//max/tests/tests/graph:ops/test_transfer": {
        "default": {
            "cpu": 2,
            "memory": 292,
        },
    },
    "//max/tests/tests/graph:ops/test_transpose": {
        "default": {
            "memory": 336,
        },
    },
    "//max/tests/tests/graph:ops/test_where": {
        "default": {
            "memory": 318,
        },
    },
    "//max/tests/tests/graph:ops/test_while_loop": {
        "default": {
            "cpu": 2,
            "memory": 314,
        },
    },
    "//max/tests/tests/graph:test_debug": {
        "default": {
            "cpu": 2,
            "memory": 368,
        },
    },
    "//max/tests/tests/graph:test_device_ref": {
        "default": {
            "cpu": 2,
            "memory": 289,
        },
    },
    "//max/tests/tests/graph:test_dialects": {
        "default": {
            "memory": 287,
        },
    },
    "//max/tests/tests/graph:test_dtype_promotion": {
        "default": {
            "cpu": 2,
            "memory": 412,
        },
    },
    "//max/tests/tests/graph:test_graph_value": {
        "default": {
            "cpu": 2,
            "memory": 361,
        },
    },
    "//max/tests/tests/graph:test_non_contiguous_tensors": {
        "default": {
            "memory": 1273,
        },
    },
    "//max/tests/tests/graph:test_shapes": {
        "default": {
            "cpu": 2,
            "memory": 289,
        },
    },
    "//max/tests/tests/graph:test_sharding_strategy": {
        "default": {
            "cpu": 2,
            "memory": 316,
        },
    },
    "//max/tests/tests/graph:test_squeeze": {
        "default": {
            "cpu": 2,
            "memory": 331,
        },
    },
    "//max/tests/tests/graph:test_tensor_value": {
        "default": {
            "cpu": 2,
            "memory": 400,
        },
    },
    "//max/tests/tests/graph:test_type": {
        "default": {
            "memory": 505,
        },
    },
    "//max/tests/tests/graph:test_type_no_context": {
        "default": {
            "memory": 293,
        },
    },
    "//max/tests/tests/graph:test_weight": {
        "default": {
            "memory": 302,
        },
    },
    "//max/tests/tests/graph:utils/test_load_gguf": {
        "default": {
            "cpu": 2,
            "memory": 294,
        },
    },
    "//max/tests/tests/graph:utils/test_load_safetensors": {
        "default": {
            "memory": 304,
        },
    },
    "//max/tests/tests/kv_cache:test_attention": {
        "default": {
            "cpu": 4,
        },
    },
    "//max/tests/tests/kv_cache:test_fp4_matmul": {
        "default": {
            "cpu": 2,
            "memory": 889,
        },
    },
    "//max/tests/tests/kv_cache:test_fp8_matmul": {
        "default": {
            "cpu": 2,
            "memory": 746,
        },
    },
    "//max/tests/tests/mojo-importer:mojo-importer": {
        "default": {
            "cpu": 3,
            "memory": 642,
        },
    },
    "//max/tests/tests/nn:test_conv": {
        "default": {
            "cpu": 3,
        },
    },
    "//max/tests/tests/nn:test_layer_norm": {
        "default": {
            "cpu": 5,
        },
    },
    "//max/tests/tests/nn:test_linear": {
        "default": {
            "cpu": 2,
        },
    },
    "//max/tests/tests/nn:test_module": {
        "default": {
            "memory": 1872,
        },
    },
    "//max/tests/tests/nn:test_rms_norm": {
        "default": {
            "cpu": 2,
            "memory": 1642,
        },
    },
    "//max/tests/tests/nn:test_sampling": {
        "default": {
            "cpu": 3,
            "memory": 6220,
        },
    },
    "//max/tests/tests/nn:test_state_dict": {
        "default": {
            "cpu": 3,
        },
    },
    "//max/tests/tests/nn:test_tensor_parallel_linear": {
        "default": {
            "cpu": 4,
        },
    },
    "//max/tests/tests/pipelines/internvl:test_embedding_merge": {
        "default": {
            "cpu": 2,
            "memory": 721,
        },
    },
    "//max/tests/tests/pipelines/internvl:test_embeddings": {
        "default": {
            "cpu": 2,
            "memory": 719,
        },
    },
    "//max/tests/tests/pipelines/lib:test_audio_generation_config": {
        "default": {
            "memory": 719,
        },
    },
    "//max/tests/tests/pipelines/lib:test_max_config_basic": {
        "default": {
            "cpu": 3,
            "memory": 704,
        },
    },
    "//max/tests/tests/pipelines/lib:test_max_config_inheritance": {
        "default": {
            "cpu": 2,
            "memory": 766,
        },
    },
    "//max/tests/tests/pipelines:test_internvl_weight_adapters": {
        "default": {
            "cpu": 2,
            "memory": 895,
        },
    },
    "//max/tests/tests/pipelines:test_parse_quant_config": {
        "default": {
            "cpu": 2,
            "memory": 895,
        },
    },
    "//max/tests/tests/profiler:tests": {
        "default": {
            "cpu": 5,
        },
    },
    "//max/tests/tests/serve/recordreplay:test_replay": {
        "default": {
            "cpu": 5,
            "memory": 144,
        },
    },
    "//max/tests/tests/serve/recordreplay:test_replay_estimation": {
        "default": {
            "cpu": 6,
        },
    },
    "//max/tests/tests/serve/scheduler:test_di": {
        "default": {
            "cpu": 2,
            "memory": 2719,
        },
    },
    "//max/tests/tests/serve/scheduler:test_paged_scheduler": {
        "default": {
            "memory": 4590,
        },
    },
    "//max/tests/tests/serve/scheduler:test_queues": {
        "default": {
            "memory": 701,
        },
    },
    "//max/tests/tests/serve/scheduler:test_scheduler": {
        "default": {
            "cpu": 2,
            "memory": 666,
        },
    },
    "//max/tests/tests/serve/scheduler:test_scheduler_config": {
        "default": {
            "cpu": 2,
            "memory": 698,
        },
    },
    "//max/tests/tests/serve/scheduler:test_scheduler_metrics": {
        "default": {
            "cpu": 2,
            "memory": 703,
        },
    },
    "//max/tests/tests/serve/scheduler:test_text_batch_constructor": {
        "default": {
            "cpu": 2,
            "memory": 697,
        },
    },
    "//max/tests/tests/serve/scheduler:test_token_budget": {
        "default": {
            "cpu": 2,
            "memory": 730,
        },
    },
    "//max/tests/tests/serve/scheduler:test_tts_scheduler": {
        "default": {
            "cpu": 2,
            "memory": 2050,
        },
    },
    "//max/tests/tests/serve:pipelines/test_audio_generator_pipeline": {
        "default": {
            "cpu": 2,
            "memory": 886,
        },
    },
    "//max/tests/tests/serve:pipelines/test_audio_generator_pipeline_sampling_params": {
        "default": {
            "cpu": 2,
            "memory": 905,
        },
    },
    "//max/tests/tests/serve:pipelines/test_stop_detection": {
        "default": {
            "cpu": 2,
            "memory": 918,
        },
    },
    "//max/tests/tests/serve:test_async_queue": {
        "default": {
            "memory": 899,
        },
    },
    "//max/tests/tests/serve:test_file_uri": {
        "default": {
            "cpu": 2,
            "memory": 868,
        },
    },
    "//max/tests/tests/serve:test_kserve_routes": {
        "default": {
            "cpu": 2,
            "memory": 912,
        },
    },
    "//max/tests/tests/serve:test_llm": {
        "default": {
            "memory": 1193,
        },
    },
    "//max/tests/tests/serve:test_lora_integration": {
        "default": {
            "cpu": 2,
            "memory": 922,
        },
    },
    "//max/tests/tests/serve:test_metrics": {
        "default": {
            "cpu": 2,
            "memory": 908,
        },
    },
    "//max/tests/tests/serve:test_multiprocessing": {
        "default": {
            "cpu": 2,
            "memory": 913,
        },
    },
    "//max/tests/tests/serve:test_openai_request": {
        "default": {
            "cpu": 2,
            "memory": 920,
        },
    },
    "//max/tests/tests/serve:test_openai_routes": {
        "default": {
            "cpu": 2,
            "memory": 1268,
        },
    },
    "//max/tests/tests/serve:test_openai_stream": {
        "default": {
            "cpu": 2,
            "memory": 1184,
        },
    },
    "//max/tests/tests/serve:test_reset_prefix_cache": {
        "default": {
            "memory": 1222,
        },
    },
    "//max/tests/tests/serve:test_routes": {
        "default": {
            "cpu": 2,
            "memory": 1175,
        },
    },
    "//max/tests/tests/serve:test_socket_settings": {
        "default": {
            "cpu": 2,
            "memory": 924,
        },
    },
    "//max/tests/tests/serve:test_stopwatch": {
        "default": {
            "cpu": 2,
            "memory": 910,
        },
    },
    "//max/tests/tests/serve:test_telemetry_worker": {
        "default": {
            "memory": 896,
        },
    },
    "//max/tests/tests/support:tests": {
        "default": {
            "memory": 481,
        },
    },
    "//max/tests/tests/torch:tests": {
        "default": {
            "cpu": 2,
            "memory": 8038,
        },
    },
    "//max/tests/tests:test_generated_dialects": {
        "default": {
            "cpu": 8,
            "memory": 2237,
        },
    },
    "//max/tests/tests:test_passes": {
        "default": {
            "cpu": 8,
            "memory": 3119,
        },
    },
    "//max/tests/tests:test_realization_context": {
        "default": {
            "cpu": 8,
            "memory": 21282,
        },
    },
    "//max/tests/tests:test_support": {
        "default": {
            "cpu": 8,
            "memory": 2274,
        },
    },
    "//max/tests/tests:test_tensor": {
        "default": {
            "cpu": 7,
            "memory": 21330,
        },
    },
}
