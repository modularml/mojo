# ===----------------------------------------------------------------------=== #
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
# ===----------------------------------------------------------------------=== #

# Verifies that the host-side accelerator-vendor checks agree with the
# `--target-accelerator` value, regardless of whether it is given as a bare
# architecture ("gfx950", "sm_90") or a vendor-prefixed form
# ("amdgpu:gfx950", "amd:gfx950", "nvidia:sm_90"). A bare target must behave
# identically to its vendor-prefixed form.
#
# These are compile-only checks: `main()` contains no device code, so every
# `comptime assert` is evaluated on the host and the build succeeds only when
# the expected vendor is detected. `--emit object` stops before linking: the
# vendor detection is a compile-time property, and linking a cross-compiled
# (for example arm64-apple-darwin) executable on the x86 Linux test host fails
# at `ld.lld`, unrelated to what is under test.

# AMD: bare arch, bare alias, and both vendor prefixes must all be AMD.
# RUN: %mojo-build --emit object --target-accelerator gfx950       -D EXPECT=amd    %s -o %t
# RUN: %mojo-build --emit object --target-accelerator mi300x       -D EXPECT=amd    %s -o %t
# RUN: %mojo-build --emit object --target-accelerator amdgpu:gfx950 -D EXPECT=amd   %s -o %t
# RUN: %mojo-build --emit object --target-accelerator amd:gfx950   -D EXPECT=amd    %s -o %t

# NVIDIA: bare arch, vendor prefix, and the generic "cuda" target must all be
# NVIDIA.
# RUN: %mojo-build --emit object --target-accelerator sm_90        -D EXPECT=nvidia %s -o %t
# RUN: %mojo-build --emit object --target-accelerator nvidia:sm_90 -D EXPECT=nvidia %s -o %t
# RUN: %mojo-build --emit object --target-accelerator cuda         -D EXPECT=nvidia %s -o %t

# Apple: bare arch and "metal:" prefix must both be Apple.
# RUN: %mojo-build --emit object --target-triple arm64-apple-darwin \
# RUN:   --target-accelerator apple-m4 -D EXPECT=apple %s -o %t
# RUN: %mojo-build --emit object --target-triple arm64-apple-darwin \
# RUN:   --target-accelerator metal:4  -D EXPECT=apple %s -o %t

# Unknown/unrecognized accelerator: none of the vendor predicates fire (the
# False-on-unknown contract). "wombat42" contains none of the vendor-relevant
# substrings, so it also guards against loose-substring false positives.
# RUN: %mojo-build --emit object --target-accelerator wombat42 -D EXPECT=none %s -o %t

from std.sys import (
    get_defined_string,
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)


def main():
    comptime expect = get_defined_string["EXPECT"]()

    comptime if expect == "amd":
        comptime assert has_amd_gpu_accelerator(), "expected an AMD accelerator"
        comptime assert (
            not has_nvidia_gpu_accelerator()
        ), "did not expect an NVIDIA accelerator"
        comptime assert (
            not has_apple_gpu_accelerator()
        ), "did not expect an Apple accelerator"
    elif expect == "nvidia":
        comptime assert (
            has_nvidia_gpu_accelerator()
        ), "expected an NVIDIA accelerator"
        comptime assert (
            not has_amd_gpu_accelerator()
        ), "did not expect an AMD accelerator"
        comptime assert (
            not has_apple_gpu_accelerator()
        ), "did not expect an Apple accelerator"
    elif expect == "apple":
        comptime assert (
            has_apple_gpu_accelerator()
        ), "expected an Apple accelerator"
        comptime assert (
            not has_amd_gpu_accelerator()
        ), "did not expect an AMD accelerator"
        comptime assert (
            not has_nvidia_gpu_accelerator()
        ), "did not expect an NVIDIA accelerator"
    elif expect == "none":
        comptime assert (
            not has_amd_gpu_accelerator()
        ), "did not expect an AMD accelerator"
        comptime assert (
            not has_nvidia_gpu_accelerator()
        ), "did not expect an NVIDIA accelerator"
        comptime assert (
            not has_apple_gpu_accelerator()
        ), "did not expect an Apple accelerator"
    else:
        comptime assert False, "unknown EXPECT value"
