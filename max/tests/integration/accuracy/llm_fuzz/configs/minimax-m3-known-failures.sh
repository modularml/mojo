##===----------------------------------------------------------------------===##
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##===----------------------------------------------------------------------===##

# shellcheck disable=SC2034  # `exclude` is consumed when this file is sourced.

# These MiniMax-M3 fuzz tests fail against MiniMax's official hosted API.
# We found that by running the same suite there, so we skip them here.
# A test that passes on the official API stays enabled.

_m3_known_failures=(
  # agentic_correctness
  'agentic_correctness:no_marker_leak_streaming_no_tools'
  'agentic_correctness:no_marker_leak_streaming_with_tools'
  # basic_reasoning_and_tool_usage
  'basic_reasoning_and_tool_usage:content-parts-thinking-off'
  'basic_reasoning_and_tool_usage:content-parts-thinking-on'
  'basic_reasoning_and_tool_usage:open-ended-thinking-off'
  'basic_reasoning_and_tool_usage:open-ended-thinking-on'
  'basic_reasoning_and_tool_usage:response-format-thinking-off'
  'basic_reasoning_and_tool_usage:response-format-thinking-on'
  'basic_reasoning_and_tool_usage:special-token-content-thinking-on'
  'basic_reasoning_and_tool_usage:tool-call-and-response-format-thinking-off'
  'basic_reasoning_and_tool_usage:tool-call-and-response-format-thinking-on'
  'basic_reasoning_and_tool_usage:tool-call-auto-thinking-off'
  'basic_reasoning_and_tool_usage:tool-call-auto-thinking-on'
  'basic_reasoning_and_tool_usage:tool-call-calculate-required-thinking-off'
  'basic_reasoning_and_tool_usage:tool-call-calculate-required-thinking-on'
  'basic_reasoning_and_tool_usage:tool-call-thinking-off'
  'basic_reasoning_and_tool_usage:tool-call-thinking-on'
  # batch_determinism
  'batch_determinism:run_to_run_determinism'
  # concurrent_stress
  'concurrent_stress:burst_then_verify'
  'concurrent_stress:concurrent_mixed_10'
  'concurrent_stress:concurrent_so_isolation'
  # endurance_soak
  'endurance_soak:endurance_steady_state_error_windows'
  'endurance_soak:endurance_streaming_leak'
  # json_schema_compliance
  'json_schema_compliance:schema_compliance_100_runs'
  # openai_spec_compliance
  'openai_spec_compliance:response_format_json_object'
  'openai_spec_compliance:response_format_json_schema'
  # openrouter_tests
  'openrouter_tests:reasoning'
  'openrouter_tests:reasoning-disabled-response-format-json-object'
  'openrouter_tests:reasoning-disabled-structured-output'
  'openrouter_tests:reasoning-disabled-tool-choice-function'
  'openrouter_tests:reasoning-disabled-tool-choice-required'
  'openrouter_tests:reasoning-enabled-response-format-json-object'
  'openrouter_tests:reasoning-enabled-structured-output'
  'openrouter_tests:reasoning-enabled-tool-choice-function'
  'openrouter_tests:reasoning-enabled-tool-choice-required'
  'openrouter_tests:response-format-json-object'
  'openrouter_tests:streaming_reasoning_tool_choice_required'
  'openrouter_tests:streaming_tool_choice_function'
  'openrouter_tests:streaming_tool_choice_required'
  'openrouter_tests:structured-output'
  'openrouter_tests:structured_json_streaming'
  'openrouter_tests:system-prompt-only'
  'openrouter_tests:tool-choice-function'
  'openrouter_tests:tool-choice-required'
  'openrouter_tests:top-logprobs'
  'openrouter_tests:verbosity-max'
  'openrouter_tests:yes-no'
  # parameter_abuse
  'parameter_abuse:max_tokens_and_max_completion_tokens_conflict'
  # pipeline_stall
  'pipeline_stall:mixed_prefill_decode_pressure'
  # so_advanced
  'so_advanced:50_property_object'
  'so_advanced:allof_three_schemas'
  'so_advanced:anyof_object_variants'
  'so_advanced:array_min_max_items'
  'so_advanced:concurrent_so_mixed'
  'so_advanced:enum_numeric'
  'so_advanced:enum_special_chars'
  'so_advanced:real_world_schema'
  'so_advanced:ref_chained'
  # so_basics
  'so_basics:allof_merge'
  'so_basics:anyof_string_int'
  'so_basics:boolean_strict'
  'so_basics:enum_large_100'
  'so_basics:enum_single_value'
  'so_basics:nested_3_levels'
  'so_basics:nullable_string'
  'so_basics:ref_simple'
  'so_basics:required_fields'
  'so_basics:simple_object'
  'so_basics:streaming_so'
  # streaming_attacks
  'streaming_attacks:cancel_after_500ms'
  # structured_output
  'structured_output:uncompilable_invalid_regex_pattern'
  'structured_output:uncompilable_unresolvable_ref'
  # tc_advanced
  'tc_advanced:so_tc_mode_switch'
  # tc_basics
  'tc_basics:single_tool_named'
  'tc_basics:single_tool_required'
  # tc_schema_enforcement
  'tc_schema_enforcement:additional_properties_false[tool_choice=required]'
  'tc_schema_enforcement:anyof_deeply_nested[tool_choice=required]'
  'tc_schema_enforcement:anyof_multi_object[tool_choice=required]'
  'tc_schema_enforcement:anyof_multiple_required_fields[tool_choice=required]'
  'tc_schema_enforcement:anyof_nullable_array[tool_choice=required]'
  'tc_schema_enforcement:anyof_nullable_boolean[tool_choice=auto]'
  'tc_schema_enforcement:anyof_nullable_boolean[tool_choice=required]'
  'tc_schema_enforcement:anyof_nullable_integer[tool_choice=auto]'
  'tc_schema_enforcement:anyof_nullable_integer[tool_choice=required]'
  'tc_schema_enforcement:anyof_nullable_number[tool_choice=required]'
  'tc_schema_enforcement:anyof_with_enum_branch[tool_choice=required]'
  'tc_schema_enforcement:anyof_with_ref_and_null[tool_choice=required]'
  'tc_schema_enforcement:ap_typed_list_dict_str_list_str[tool_choice=auto]'
  'tc_schema_enforcement:ap_typed_list_dict_str_list_str[tool_choice=required]'
  'tc_schema_enforcement:ap_typed_pydantic_terminate_eval[tool_choice=auto]'
  'tc_schema_enforcement:ap_typed_pydantic_terminate_eval[tool_choice=required]'
  'tc_schema_enforcement:enum_with_object_value[tool_choice=required]'
  'tc_schema_enforcement:integer_scientific_notation[tool_choice=auto]'
  'tc_schema_enforcement:integer_scientific_notation[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_additional_props_false[tool_choice=auto]'
  'tc_schema_enforcement:ref_defs_additional_props_false[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_adversarial_all_constraints[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_anyof_nullable[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_array_enforcement[tool_choice=auto]'
  'tc_schema_enforcement:ref_defs_array_enforcement[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_array_of_enums[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_boolean_type[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_chain_a_b_c[tool_choice=auto]'
  'tc_schema_enforcement:ref_defs_chain_a_b_c[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_cross_referencing[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_enum_enforced[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_fail_open[tool_choice=auto]'
  'tc_schema_enforcement:ref_defs_fail_open[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_integer_enforcement[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_nested_array_in_object[tool_choice=auto]'
  'tc_schema_enforcement:ref_defs_nested_array_in_object[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_recursive[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_recursive_enum_leaf[tool_choice=auto]'
  'tc_schema_enforcement:ref_defs_recursive_enum_leaf[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_required_only_minimal[tool_choice=auto]'
  'tc_schema_enforcement:ref_defs_required_only_minimal[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_reused_same_def[tool_choice=required]'
  'tc_schema_enforcement:ref_defs_type_array_nullable[tool_choice=required]'
  'tc_schema_enforcement:type_list_array_with_items[tool_choice=required]'
  'tc_schema_enforcement:type_list_object_with_properties[tool_choice=auto]'
  'tc_schema_enforcement:type_list_object_with_properties[tool_choice=required]'
  # tc_streaming_protocol
  'tc_streaming_protocol:finish_reason_is_tool_calls'
  # tool_arguments_json_stability
  'tool_arguments_json_stability:forced_search_args_json_32_runs'
  # tool_calling
  'tool_calling:interleaved_thinking_multi_section'
  'tool_calling:interleaved_thinking_multi_section_streaming'
  'tool_calling:malformed_tool_name_special_chars'
  'tool_calling:required_no_content_leak'
  'tool_calling:tool_schema_with_oneof_and_const'
  # tool_schema_validation
  'tool_schema_validation:enforce_format_email'
  'tool_schema_validation:enforce_max_length'
  'tool_schema_validation:enforce_min_properties'
  'tool_schema_validation:enforce_missing_required'
  'tool_schema_validation:enforce_nested_wrong_type'
  'tool_schema_validation:enforce_not_forbidden'
  'tool_schema_validation:enforce_number_over_maximum'
  'tool_schema_validation:enforce_object_as_array'
  'tool_schema_validation:enforce_string_pattern'
  'tool_schema_validation:enforce_wrong_type'
  'tool_schema_validation:tool_choice_specific_schema'
)

# Join into the comma-separated form the --exclude flag expects.
exclude=$(IFS=,; printf "%s" "${_m3_known_failures[*]}")
