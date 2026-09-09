# Add `mmap` Module

**Status**: Proposed.

This picks up the discussion from #1134 and the earlier proposal in #3218,
incorporating the review feedback from the stdlib team.

## Problem

Mojo has no way to memory-map files or allocate anonymous mapped regions.
Users who need this today must manually call `external_call["mmap", ...]`,
check for `MAP_FAILED`, remember to `munmap` on every exit path, and handle
the platform constant differences between Linux and macOS.

```mojo
# What users have to write today:
var ptr = external_call["mmap", UnsafePointer[UInt8, MutExternalOrigin]](
    UnsafePointer[UInt8, MutExternalOrigin](),
    length, c_int(0x1), c_int(0x02), c_int(fd), Int64(0),
)
```

## Prior Discussion

The original feature request (#1134) and @KCaverly's proposal (#3218) were
reviewed by the stdlib team. The feedback was:

- Mode constants should be **numeric bit-flags**, not strings — they need to
  be OR-able to match POSIX semantics.
- Include `PROT_NONE`.
- Avoid "stringly typed" APIs.

This proposal addresses all three points. Plain `comptime Int` constants are
used (matching the `O_CREAT` / `O_TRUNC` pattern in `io/file.mojo`) and raw
`prot`/`flags` arguments are accepted rather than a mode enum.

@martinvuyk's suggestion of folding mmap into `FileHandle` as
`open[memory_mapped=True]()` was also considered. A separate module is
cleaner: mmap supports anonymous mappings (no file at all), the lifetime of
a mapping is independent of the fd, and the access pattern (`Span` over
bytes) is fundamentally different from `FileHandle`'s read/write/seek API.

## Proposal

Two layers, following the existing stdlib pattern (e.g. thin wrappers in
`sys/_libc.mojo` used internally by `FileHandle`, `Process`, etc.):

1. **Low-level thin wrappers** in `sys/_libc.mojo` — `@always_inline`
   functions around `mmap`, `munmap`, `msync`, `mprotect`, and `madvise`.
   Private to the stdlib, matching the existing `close()`, `write()`,
   `pipe()`, etc. in the same file.

2. **Public RAII API** in a new `std/mmap` package — `MmapRegion` struct
   and `mmap_file()` convenience function.

### `MmapRegion`

A move-only struct that owns a mapped region and unmaps it on destruction,
following the same pattern as `FileHandle`:

```mojo
struct MmapRegion(Movable, Sized):
    var _ptr: UnsafePointer[UInt8, MutExternalOrigin]
    var _length: Int

    fn __init__(out self, *, length, prot, flags, fd, offset) raises
    fn __del__(deinit self)           # munmap, errors silenced
    fn close(mut self) raises         # munmap, errors raised
    fn as_bytes(ref self) -> Span[UInt8, __origin_of(self)]
    fn sync(self, flags: Int = MS_SYNC) raises   # msync
    fn protect(self, prot: Int) raises            # mprotect
    fn __len__(self) -> Int
    fn __enter__(var self) -> Self
```

Design choices:

- **Move-only** — no `__copyinit__`, prevents double-munmap.
- **`as_bytes()` returns `Span`** — bounded access without exposing raw
  pointers in the public API.
- **Accepts raw `fd: Int`** rather than `FileHandle` — supports anonymous
  mappings (`fd=-1`) and fds obtained from other sources.

### `mmap_file()`

Covers the common case of mapping an entire file:

```mojo
fn mmap_file(path: String, *, writable: Bool = False) raises -> MmapRegion
```

Stats the file for size, opens it, maps it, closes the fd, and returns the
region.

### Constants

Standard POSIX constants (`PROT_*`, `MAP_SHARED`, `MAP_PRIVATE`, `MAP_FIXED`,
`MS_*`), plus Linux-specific flags (`MAP_HUGETLB`, `MAP_LOCKED`,
`MAP_POPULATE`, etc.) and `MADV_*` constants for `madvise`.

Platform-differing values use `platform_map`. Linux-only flags use
`platform_map` with only `linux=` specified (compile error on macOS),
following the pattern in `sys/_libc_errno.mojo` (e.g. `ECHRNG`,
`EL2NSYNC`).

## Examples

Map a file for reading:

```mojo
from mmap import mmap_file

with mmap_file("/path/to/data.bin") as region:
    var data = region.as_bytes()
    print("size:", len(region), "first byte:", data[0])
```

Anonymous scratch region:

```mojo
from mmap import MmapRegion, PROT_READ, PROT_WRITE, MAP_PRIVATE, MAP_ANONYMOUS

var region = MmapRegion(
    length=4096,
    prot=PROT_READ | PROT_WRITE,
    flags=MAP_PRIVATE | MAP_ANONYMOUS,
    fd=-1,
    offset=0,
)
var buf = region.as_bytes()
buf[0] = 42
```

## Alternatives Considered

- **Python-style `mmap.mmap` with `read()`/`write()`/`seek()`**: Higher-level
  than needed for a stdlib primitive. Can be built on top of `MmapRegion`.
- **Folding into `FileHandle`**: mmap's lifetime and access model are too
  different from fd-based I/O.
- **Requiring `FileHandle` instead of raw fd**: Would prevent anonymous
  mappings and limit flexibility when the caller already has an fd.
- **Exposing raw syscall wrappers as public API**: The stdlib keeps thin libc
  wrappers private (`sys/_libc.mojo`), matching Rust's `memmap2`, Go's
  `syscall.Mmap`, and Python's `mmap` — all use all-or-nothing RAII.
