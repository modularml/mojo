# Adding a new GPU target

This guide explains how to add support for a new GPU architecture to the Mojo
standard library. Targets live in
[`std/_gpu/host/info.mojo`](../../../stdlib/std/_gpu/host/info.mojo); this
document is for contributors extending that file and is not part of the public
API.

When adding support for a new GPU architecture, you must create a target
configuration function that returns a `_TargetType`. The sections below explain
the components of the MLIR target configuration, with special focus on the
`data_layout` string.

## MLIR target components

Each GPU target function returns an MLIR `kgen.target` attribute with these
fields:

- **triple**: Target triple (e.g., "nvptx64-nvidia-cuda", "amdgcn-amd-amdhsa").
- **arch**: Architecture name (e.g., "sm_80", "gfx942", "apple-m4").
- **features**: Target-specific features (e.g., "+ptx81,+sm_80").
- **tune_cpu**: Optimization target (usually same as arch, can differ for
  tuning).
- **data_layout**: LLVM data layout string (explained in detail below).
- **index_bit_width**: Bit width for index types (usually 64).
- **simd_bit_width**: SIMD register width (usually 128 for modern GPUs).

## Understanding data layout strings

The `data_layout` string describes memory layout characteristics for the target
architecture. It follows LLVM's data layout specification format:
<https://llvm.org/docs/LangRef.html#data-layout>, and is used by the compiler
to make decisions about memory access patterns, type layouts, and
optimizations.

### Format overview

The string consists of specifications separated by dashes (`-`):

- **Endianness**: `e` (little-endian) or `E` (big-endian).
- **Pointers**: `p[addr_space]:size:abi:pref:idx`.
- **Integers**: `i<size>:<abi>:<pref>`.
- **Floats**: `f<size>:<abi>:<pref>`.
- **Vectors**: `v<size>:<abi>:<pref>`.
- **Native widths**: `n<size>:<size>:...`.
- **Stack alignment**: `S<size>`.
- **Address space**: `A<number>`.
- **Mangling**: `m:<style>` (e.g., `m:e` for ELF).

### Component details

#### Endianness

- `e`: Little-endian (all modern GPUs use this).
- `E`: Big-endian (rarely used).

#### Pointer specifications: `p[addr_space]:size:abi:pref:idx`

Defines pointer sizes and alignments for different memory spaces:

- **Address space**: Optional number (0-9) specifying memory type:
  - `p` or `p0`: Generic/flat address space.
  - `p1`: Global memory (AMD) or device memory.
  - `p2`: Constant memory (AMD).
  - `p3`: Shared/local memory (NVIDIA) or local memory (AMD).
  - `p4`: Constant memory (NVIDIA) or generic memory (AMD).
  - `p5`: Local/private memory (NVIDIA/AMD).
  - `p6-p9`: Vendor-specific address spaces.
- **size**: Pointer size in bits.
- **abi**: ABI-required alignment in bits.
- **pref**: Preferred alignment in bits (optional).
- **idx**: Index type size in bits (optional).

Examples:

- `p3:32:32` means shared memory uses 32-bit pointers with 32-bit alignment.
- `p:64:64:64` means generic pointers are 64 bits with 64-bit alignment.
- `p7:160:256:256:32` means address space 7 uses 160-bit pointers with 256-bit
  alignment.

#### Integer specifications: `i<size>:<abi>:<pref>`

Defines alignment for integer types:

- **size**: Integer size in bits (1, 8, 16, 32, 64, 128, 256, etc.).
- **abi**: Minimum ABI alignment in bits.
- **pref**: Preferred alignment in bits (optional, defaults to abi).

Examples:

- `i64:64` means 64-bit integers have 64-bit alignment.
- `i128:128` means 128-bit integers have 128-bit alignment.
- `i1:8:8` means 1-bit booleans are stored in 8-bit aligned bytes.

#### Float specifications: `f<size>:<abi>:<pref>`

Similar to integers but for floating-point types:

Examples:

- `f32:32:32` means 32-bit floats have 32-bit alignment.
- `f64:64:64` means 64-bit doubles have 64-bit alignment.

#### Vector specifications: `v<size>:<abi>:<pref>`

Defines alignment for vector types:

- **size**: Vector size in bits.
- **abi**: ABI alignment in bits.
- **pref**: Preferred alignment in bits (optional).

Examples:

- `v16:16` means 16-bit vectors aligned to 16 bits.
- `v128:128:128` means 128-bit vectors have 128-bit alignment.

#### Native integer widths: `n<size>:<size>:...`

Specifies which integer widths are "native" (efficient) for the target. The
compiler will prefer these sizes for operations.

Examples:

- `n16:32:64` means 16, 32, and 64-bit operations are efficient.
- `n32:64` means 32 and 64-bit operations are efficient.
- `n8:16:32` means 8, 16, and 32-bit operations are efficient.

#### Stack alignment: `S<size>`

Specifies natural stack alignment in bits.

Example: `S32` means 32-bit stack alignment.

#### Address space: `A<number>`

Specifies the default address space for allocations.

Example: `A5` means use address space 5 by default.

#### Non-integral pointers: `ni:<space>:<space>:...`

Lists address spaces where pointers cannot be cast to integers.

Example: `ni:7:8:9` means address spaces 7, 8, and 9 have non-integral
pointers.

## Vendor-specific patterns

### NVIDIA GPUs (CUDA/PTX)

Typical data layout for NVIDIA GPUs (sm_60 and later):

```text
e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64
```

Breakdown:

- `e`: Little-endian.
- `p3:32:32`: Shared memory pointers are 32-bit.
- `p4:32:32`: Constant memory pointers are 32-bit.
- `p5:32:32`: Local memory pointers are 32-bit.
- `p6:32:32`, `p7:32:32`: NVIDIA-specific address spaces.
- `i64:64`, `i128:128`, `i256:256`: Integer alignments.
- `v16:16`, `v32:32`: Vector alignments for warp operations.
- `n16:32:64`: Native integer widths (16, 32, and 64-bit operations).

Note: NVIDIA GPUs use address-space-specific 32-bit pointers for shared,
constant, and local memory, while the default address space (not specified)
uses 64-bit pointers. This matches the PTX memory model.

### AMD GPUs (ROCm/HIP)

Typical data layout for AMD GPUs (CDNA and RDNA):

```text
e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9
```

AMD GPUs use more address spaces and have more complex specifications:

- `m:e`: ELF mangling style.
- `p:64:64`: Default pointers are 64-bit (unified addressing).
- `p1:64:64`: Global memory uses 64-bit pointers.
- `p2:32:32`: Constant memory uses 32-bit pointers.
- `p3:32:32`: Local/shared memory uses 32-bit pointers.
- `p4:64:64`: Generic address space uses 64-bit pointers.
- `p5:32:32`: Private memory uses 32-bit pointers.
- `p7`, `p8`, `p9`: Complex buffer descriptors (160, 128, 192 bits).
- Extensive vector sizes (`v16` through `v2048`) for wavefront operations.
- `n32:64`: Native integer widths.
- `S32`: 32-bit stack alignment.
- `A5`: Default address space is 5.
- `G1`: Global address space is 1.
- `ni:7:8:9`: Address spaces 7, 8, 9 have non-integral pointers.

### Apple Metal GPUs

Typical data layout for Apple Silicon:

```text
e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32
```

Apple GPUs have unified memory architecture:

- `p:64:64:64`: 64-bit pointers with explicit preferred alignment (unified
  memory).
- Explicit specifications for all integer sizes (`i1`, `i8`, `i16`, `i32`,
  `i64`).
- Explicit float alignments (`f32:32:32`, `f64:64:64`).
- Comprehensive vector size coverage (`v16` through `v1024`).
- `n8:16:32`: Native integer widths (8, 16, and 32-bit operations).

## How to obtain data layout strings

When adding support for a new GPU architecture, obtain the data layout string
using these methods:

### Method 1: Query LLVM/Clang (recommended)

Use Clang to query the target's default data layout:

For NVIDIA GPUs:

```bash
echo 'target triple = "nvptx64-nvidia-cuda"' > test.ll
clang -S test.ll -o - | grep datalayout
```

For AMD GPUs:

```bash
echo 'target triple = "amdgcn-amd-amdhsa"' > test.ll
clang -S test.ll -o - | grep datalayout
```

### Method 2: Consult LLVM source code

Check the LLVM source for target data layout definitions:

- **NVIDIA**: `llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp` (see
  `computeDataLayout()`).
- **AMD**: `llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp` (see
  `getGPUDataLayout()`).

### Method 3: Reference similar GPUs

For GPUs in the same architecture family, the data layout is often identical:

- All NVIDIA Ampere/Ada/Hopper GPUs (sm_80+) use the same data layout.
- AMD CDNA GPUs share similar layouts.
- Apple Metal GPUs have consistent patterns across generations.

When in doubt, use the data layout from a GPU in the same family.

### Method 4: Consult vendor documentation

Refer to official programming guides and specifications:

- **NVIDIA**:
  [LLVM NVPTX Usage Guide](https://llvm.org/docs/NVPTXUsage.html#data-layout),
  CUDA Programming Guide, PTX ISA documentation.
- **AMD**: ROCm documentation, LLVM AMDGPU documentation.
- **Apple**: Metal Programming Guide, Metal Shading Language Specification.

The LLVM NVPTX documentation recommends this data layout for 64-bit NVIDIA
GPUs:

```text
e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64
```

Note: The data layouts in `info.mojo` use address-space-specific pointer
specifications (p3, p4, p5, etc.) rather than the generic `p:64:64:64`. This
provides more precise control over memory access patterns for different memory
spaces in GPU kernels.

## Field-by-field explanation

### Triple

The target triple identifies the architecture, vendor, and operating system:

- **NVIDIA**: `nvptx64-nvidia-cuda` (64-bit) or `nvptx-nvidia-cuda` (32-bit).
- **AMD**: `amdgcn-amd-amdhsa` (HSA runtime).
- **Apple**: `air64-apple-macosx` (Metal on macOS).

### Arch

The architecture name specifies the GPU generation:

- **NVIDIA**: `sm_XX` where XX is the compute capability (e.g., `sm_80` for
  compute 8.0).
  - Find compute capability at <https://developer.nvidia.com/cuda-gpus>.
  - Format: `sm_XY` maps to compute capability `X.Y`, `sm_XYZ` maps to `XY.Z`.
- **AMD**: `gfxXXXX` where XXXX is the GFX version (e.g., `gfx942` for
  MI300X).
  - Find GFX version in ROCm documentation or GPU specifications.
- **Apple**: `apple-mX` where X is the chip generation (e.g., `apple-m4`).

### Features

Target-specific features enabled for code generation:

- **NVIDIA**: `+ptxXX,+sm_YY` where XX is PTX version and YY is compute
  capability.
  - PTX version should match your CUDA toolkit version (see PTX ISA docs).
  - Example: `+ptx85,+sm_90a` enables PTX 8.5 and compute 9.0a features.
  - **Q: Is specifying PTX version redundant?** A: No, PTX version determines
    available instructions and features, independent of compute capability.
- **AMD**: Often empty (`""`) as features are implied by architecture.
- **Apple**: Often empty (`""`) for Metal GPUs.

### Tune CPU

Specifies the optimization target for code generation:

- Usually the same as `arch` (e.g., `tune_cpu = "sm_90a"`).
- Can differ if you want to optimize for a different microarchitecture while
  maintaining compatibility (e.g., `arch = "sm_80"`, `tune_cpu = "sm_90a"`).
- Some older GPU entries omit this field (see GTX 970, GTX 1080 Ti).

### Index bit width

The bit width for index types used in address calculations:

- **32-bit systems**: `index_bit_width = 32`.
- **64-bit systems**: `index_bit_width = 64`.
- Most modern GPUs use 64-bit indexing for large memory spaces.

### SIMD bit width

The width of SIMD registers in bits:

- **Modern GPUs**: Usually `simd_bit_width = 128` (128-bit vector operations).
- This represents the native vector width for efficient operations.
- **How to find this**: Based on warp/wavefront width and register
  architecture:
  - NVIDIA: 128 bits (4 x 32-bit values per warp operation).
  - AMD: 128 bits for CDNA/RDNA architectures.
  - Apple: 128 bits for Metal GPUs.

## Step-by-step guide for adding a new GPU

Follow these steps to add support for a new GPU architecture:

### Step 1: Gather GPU information

Collect these specifications for your GPU:

- **Model name**: e.g., "H100", "MI300X", "M4".
- **Compute capability** (NVIDIA) or **GFX version** (AMD) or **Metal
  version** (Apple).
- **Architecture family**: Identify the family (e.g., Hopper, CDNA3, Apple M
  series).
- **SM/CU count**: Number of streaming multiprocessors or compute units.
- **Target triple**: Standard LLVM triple for the vendor.
- **Data layout string**: Obtain using methods described above.

To find SM count for NVIDIA GPUs, use this CUDA code:

```c
void printMultiProcessorCount() {
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
}
```

Or check vendor specifications:

- **NVIDIA**: <https://developer.nvidia.com/cuda-gpus>.
- **AMD**: ROCm device specifications.

### Step 2: Create the target function

Add a new function that returns the MLIR target configuration.

Example for an NVIDIA GPU:

```mojo
def _get_your_gpu_target() -> _TargetType:
    """Creates an MLIR target configuration for Your GPU.

    Returns:
        MLIR target configuration for Your GPU.
    """
    return __mlir_attr[
        `#kgen.target<triple = "nvptx64-nvidia-cuda", `,
        `arch = "sm_90a", `,
        `features = "+ptx85,+sm_90a", `,
        `tune_cpu = "sm_90a", `,
        `data_layout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64",`,
        `index_bit_width = 64,`,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
```

Place this function with other GPU target functions in `info.mojo` (search for
`_get_*_target()` functions).

### Step 3: Create the `GPUInfo` alias

Define the GPU characteristics using the appropriate architecture family:

```mojo
comptime YourGPU = GPUInfo.from_family(
    family=NvidiaHopperFamily,  # Choose the appropriate family
    name="Your GPU",
    api="cuda",  # Identifies the vendor: "cuda", "hip", or "metal"
    arch_name="hopper",
    compute=9.0,  # Must match arch (9.0 -> sm_90, 12.1 -> sm_121)
    version="sm_90a",
    sm_count=132,  # Number of streaming multiprocessors
)
```

Place this alias with other GPU aliases in `info.mojo`.

### Step 4: Update `_get_info_from_target`

Add your architecture to the constraint list in the `_get_info_from_target`
function:

```mojo
comptime assert StaticString(target_arch)
    in (
        # NVIDIA
        StaticString("cuda"),
        StaticString("52"),
        StaticString("90a"),  # Add your architecture here
        # ... rest of architectures ...
    ), String("the target architecture '",
    target_arch0,
    "' is invalid or not currently supported")
```

Then add the mapping in the `comptime` block:

```mojo
comptime if target_arch == "52":
    return materialize[GTX970]()
elif target_arch == "90a":  # Add your mapping here
    return materialize[YourGPU]()
# ... rest of mappings ...
```

Note: The `target_arch` has the `sm_` prefix stripped, so `sm_90a` becomes
`90a`.

Note: GPUs are currently 1:1 with the `target_arch` string. This is going to
be changed to support multiple GPUs per `target_arch` in the future.

### Step 5: Update `GPUInfo.target` method

Add the target mapping in the `target()` method of the `GPUInfo` struct:

```mojo
def target(self) -> _TargetType:
    """Gets the MLIR target configuration for this GPU.

    Returns:
        MLIR target configuration for the GPU.
    """
    if self.name == "NVIDIA Tesla P100":
        return _get_teslap100_target()
    if self.name == "Your GPU":  # Add your GPU here
        return _get_your_gpu_target()
    # ... rest of mappings ...
```

### Step 6: Build and test

Build the standard library to verify your changes:

```bash
./bazelw build //Mojo/stdlib/std
```

Test with a simple GPU program:

```bash
MODULAR_MOJO_MAX_IMPORT_PATH=bazel-bin/Mojo/stdlib/std mojo your_test.mojo
```

Run existing GPU tests to ensure nothing broke:

```bash
./bazelw test //Mojo/stdlib/test/gpu/...
```

## Common pitfalls

Avoid these common mistakes when adding GPU support:

1. **Mismatched compute capability**: Ensure `compute` matches `arch` (e.g.,
   `compute=9.0` with `arch="sm_90a"`).
2. **Incorrect pointer sizes**: Verify address space pointer sizes match
   hardware capabilities.
3. **Missing vector alignments**: Include all vector sizes your kernels will
   use.
4. **Wrong endianness**: All modern GPUs are little-endian (use `e`).
5. **Inconsistent with LLVM**: Data layout must match LLVM's target
   definition.
6. **Copy-paste errors**: Double-check field values when adapting from similar
   GPUs.
7. **Forgetting to update all 5 locations**: Target function, alias,
   constraint list, `comptime` block, and `target()` method.
8. **PTX/driver version mismatch**: Ensure PTX version is supported by your
   CUDA driver.

## Validation checklist

Before submitting your GPU addition:

- [ ] Target function created and documented.
- [ ] `GPUInfo` alias defined with correct family.
- [ ] Architecture added to constraint list in `_get_info_from_target`.
- [ ] Mapping added to `comptime` block in `_get_info_from_target`.
- [ ] Mapping added to `GPUInfo.target()` method.
- [ ] Data layout string validated against LLVM documentation.
- [ ] Compute capability matches architecture name.
- [ ] SM/CU count verified against official specifications.
- [ ] Standard library builds successfully.
- [ ] Existing tests pass.
- [ ] Manual testing with simple GPU kernel.

## Related files

- **`std/sys/info.mojo`**: Defines `_TargetType` as `!kgen.target` and the
  `CompilationTarget` struct.
- **LLVM Documentation**: <https://llvm.org/docs/LangRef.html#data-layout>
  (complete data layout specification).
- **LLVM NVPTX Usage**: <https://llvm.org/docs/NVPTXUsage.html>
  (NVIDIA-specific guidance).

## Examples in `info.mojo`

See real-world examples by searching for these functions:

- `_get_h100_target()`: NVIDIA Hopper H100 (compute 9.0).
- `_get_mi250x_target()`: AMD CDNA2 MI250X.
- `_get_mi300x_target()`: AMD CDNA3 MI300X.
- `_get_metal_m4_target()`: Apple Metal M4.
- `_get_metal_m4_metal4_target()`: Apple Metal M4 with Metal 4.0.
- `_get_rtx5090_target()`: NVIDIA Blackwell consumer GPU.

Each example demonstrates the complete target configuration for that GPU
family.
