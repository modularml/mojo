# Logging library

Use this library to emit log messages from any layer of the stack to a file or
stdout, with timestamps and severity levels.

## Interface

The logger generally expects a single formatting string and a list of arguments
to be formatted within it. The format string uses `fmt` syntax, which is
similar to `std::format`.

### C++

`MLOG_DEBUG`, `MLOG_INFO`, `MLOG_WARN`, `MLOG_ERROR`, and `MLOG_FATAL`
are convenience wrappers around `MLOG(level, "format string", args...)`, which
emits a message at the specified level to a file or stdout. `MLOG_FATAL` will
abort the user program after logging the message.

#### Structured key-value records

`MLOG_KV(level, key, value, ...)` emits named fields instead of a formatted
message. It takes alternating key-value pairs, up to four pairs, and the keys
must be strings:

```cpp
MLOG_KV(LogLevel::INFO,
        "event",       "span_start",
        "operation",   "prefill",
        "batch_id",    batchId,
        "request_id",  requestId);
```

In JSON mode each pair becomes a top-level field, which makes the values
usable as indexed facets downstream. Otherwise the pairs render as
`key=value` tokens after the usual prefix:

```text
[INFO] event=span_start operation=prefill batch_id=42 request_id=a1b2c3
```

The same record under `MODULAR_LOG_JSON`:

```json
{"timestamp": "2026-03-16T12:00:00.123456Z", "level": "INFO",
 "channel": "default", "event": "span_start", "operation": "prefill",
 "batch_id": 42, "request_id": "a1b2c3"}
```

Two properties differ from `MLOG`. Arguments are not evaluated at all when the
level is filtered out, so `MLOG_KV` is safe to place on hot paths. And keys are
written verbatim, so avoid `timestamp`, `level`, and `channel` — a key that
collides with an envelope field produces a duplicate JSON key.

Keep keys to 16 characters or fewer. A longer key does not fit `LogArg`'s
inline buffer, so it is copied into the record's shared 256-byte arena along
with every other string in that record — and the arena clips rather than
grows. A clipped key is a silently renamed field, and two long keys sharing a
prefix can clip to the same name.

##### More examples

Values keep their type through to JSON, so numbers and booleans arrive
unquoted and stay filterable as numbers rather than strings:

```cpp
MLOG_KV(LogLevel::INFO,
        "event",       "cache_lookup",
        "hit",         found,           // bool    -> true
        "latency_ms",  elapsedMs,       // double  -> 1.5
        "entries",     cache.size());   // integer -> 4096
```

```json
{"timestamp": "...", "level": "INFO", "channel": "default",
 "event": "cache_lookup", "hit": true, "latency_ms": 1.5, "entries": 4096}
```

On a hot path there is no need to guard the call yourself. The macro checks
the level before it evaluates anything, so an expensive argument costs nothing
when the record is filtered out:

```cpp
// summarize() does not run unless DEBUG is enabled.
MLOG_KV(LogLevel::DEBUG, "event", "batch_done", "stats", summarize(batch));
```

Four pairs is the ceiling, and the count must be even. Both are compile-time
errors, as is a key that is not a string:

```cpp
MLOG_KV(LogLevel::INFO, "a", 1, "b");            // error: needs pairs
MLOG_KV(LogLevel::INFO, 7, "value");             // error: key must be a string
MLOG_KV(LogLevel::INFO, "a", 1, "b", 2,
        "c", 3, "d", 4, "e", 5);                 // error: at most four pairs
```

Emit a second record when a call site needs more than four fields; there is no
continuation form.

#### Timing a scope

`SpanGuard`, from `Support/SpanGuard.h`, times a scope and reports it as a pair
of `MLOG_KV` records sharing a `span_id`:

```cpp
{
  M::Log::SpanGuard span("prefill");
  runPrefill();
}
```

```text
event=span_start operation=prefill span_id=802754119...
event=span_end operation=prefill span_id=802754119... duration_us=1423
```

The end record is emitted from the destructor, so an early return still closes
the span. Durations come from `steady_clock`, so a wall-clock adjustment
mid-span cannot skew them. The `operation` string is stored, not copied, so it
must outlive the guard — pass a literal.

Span ids are drawn per thread from a random 64-bit base, so they are distinct
across threads without a shared counter. Records logged inside the scope do not
pick up the `span_id` automatically; pass `span.getSpanId()` explicitly if a
record needs to join the span.

### Mojo

There is a Mojo interface wrapping the C++ Log library. It uses the same `fmt`
formatting underneath, so the same log message will produce the same output
regardless of source (modulo timestamps etc.). The interface specifically is:

```mojo
mlog["format string here: {}", LogLevel.INFO]("arguments here")
mlog_info["all {} log convenience functions work"](5)
```

Arguments are captured and transformed into a form suitable for the FFI call,
which is the LogArg class on the C++ side of the implementation.

## Environment variables

The following environment variables control logging behavior.

| Variable                   | Description                                                           |
|----------------------------|-----------------------------------------------------------------------|
| `MODULAR_LOG_STDOUT`       | `false` suppresses stdout output. Default true. See note on sinks.    |
| `MODULAR_LOG_FILE`         | Path to the log file. If unset, no file is written.                   |
| `MODULAR_LOG_ISO_TIME`     | Output timestamps in `YYYY-MM-DD:hh:mm:ss` format.                    |
| `MODULAR_LOG_LEVEL`        | Minimum message level to write. Corresponds to the macro names above. |
| `MODULAR_LOG_MICROSECONDS` | Include microseconds in the timestamp.                                |
| `MODULAR_LOG_NO_ENHANCED`  | Disable all prefix formatting, including the level and timestamp.     |
| `MODULAR_LOG_NO_TIMESTAMP` | Disable the timestamp while keeping the level prefix.                 |
| `MODULAR_LOG_JSON`         | Output JSON log lines, overriding other output configurations.        |
| `MODULAR_LOG_NO_SUMMARY`   | Suppress the shutdown summary printed when the process exits.         |

## Output sinks

Output can be sent to stdout (`MODULAR_LOG_STDOUT` is true) or to a file
(`MODULAR_LOG_FILE` is set to some valid path). These options are orthogonal;
if both are set, output goes to both, and if set to `false` and `""` (empty
string or unset) then the logging is effectively turned off.

## Async logging

Log calls are non-blocking. Each call serialises the record into a lock-free
MPSC ring buffer and returns immediately; a dedicated consumer thread reads from
the buffer and writes to the configured sinks. This means log output may appear
slightly after the call site executes, and sink writes are batched — they are
flushed to the OS when the ring drains rather than after every record.

### Dropped records

The ring buffer has a fixed capacity. If producers enqueue records faster than
the consumer can drain them, new records are dropped rather than blocking the
caller. This is intentional: logging must never slow down or stall the work
being observed.

When the process exits, a summary line is printed to stdout if any records were
written or dropped during the process lifetime:

```text
[Logger] shutdown: 142000 records written, 0 dropped
```

A nonzero drop count indicates the log rate exceeded consumer throughput. Set
`MODULAR_LOG_NO_SUMMARY` to suppress this line.

### String argument lifetime

String arguments are copied into a per-slot arena at enqueue time so they remain
valid after the call returns. Each slot holds up to 256 bytes of string data. If
the total string content in a single log record exceeds 256 bytes, the excess is
silently clipped. Keep string arguments short.

What decides whether a string is copied is its length, not its storage
duration: values of 16 bytes or fewer are stored inline in the `LogArg` itself
and never reach the arena, while anything longer is copied — a string literal
included.

## JSON output format

When `MODULAR_LOG_JSON` is set, each log line is a self-contained JSON object
followed by a newline (newline-delimited JSON / NDJSON). Other formatting flags
(`MODULAR_LOG_ISO_TIME`, `MODULAR_LOG_NO_TIMESTAMP`, etc.) are ignored in this
mode.

A line is one of two shapes. Records from `MLOG` carry a `message` field and
nothing else; records from `MLOG_KV` carry their pairs as additional
top-level fields and have no `message`.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["timestamp", "level", "channel"],
  "properties": {
    "timestamp": {
      "type": "string",
      "description": "UTC time in ISO 8601 format with microsecond precision.",
      "examples": ["2026-03-16T12:00:00.123456Z"]
    },
    "level": {
      "type": "string",
      "enum": ["DBG", "INFO", "WARN", "ERR", "FATL"],
      "description": "Severity level of the log message."
    },
    "channel": {
      "type": "string",
      "description": "Name of the channel the record was logged on."
    },
    "message": {
      "type": "string",
      "description": "Log message text. Present only on MLOG records."
    }
  },
  "oneOf": [
    {
      "required": ["message"],
      "additionalProperties": false
    },
    {
      "not": {"required": ["message"]},
      "minProperties": 4,
      "additionalProperties": {
        "type": ["string", "number", "boolean"],
        "description": "One MLOG_KV pair. Up to four are present."
      }
    }
  ]
}
```
