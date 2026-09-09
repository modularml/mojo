# MiniMax M3 `reasoning_split` reference captures

Reference responses captured from the official MiniMax endpoint
(`https://api.minimax.io/v1/chat/completions`, model `MiniMax-M3`) to document
the `reasoning_split` parameter behavior implemented for CENG-592.

Prompt: `What is 17 multiplied by 24? Think step by step, then give the final
answer.` with `temperature=0`.

| File                                        | Setting                                         |
|---------------------------------------------|-------------------------------------------------|
| `official_reasoning_split_true.json`        | non-streaming, `reasoning_split=true` (default) |
| `official_reasoning_split_false.json`       | non-streaming, `reasoning_split=false`          |
| `official_stream_reasoning_split_true.sse`  | streaming, `reasoning_split=true`               |
| `official_stream_reasoning_split_false.sse` | streaming, `reasoning_split=false`              |

Key behavior with `reasoning_split=false`: the thinking text is folded back into
`content` as `<think>\n{thinking}\n</think>\n\n{answer}` and no separate
`reasoning_content` / `reasoning_details` field is returned.
