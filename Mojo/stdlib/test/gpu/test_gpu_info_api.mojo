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

from std._gpu.host.info import GPUInfo, _all_targets
from std.sys.info import Vendor, _vendor_from_arch
from std.testing import TestSuite, assert_equal


comptime _MI250X_SPELLINGS = (
    StaticString("gfx90a"),
    StaticString("mi250x"),
    StaticString("amdgpu:gfx90a"),
    StaticString("amd:gfx90a"),
)


def _documented_api(vendor: Vendor) raises -> StaticString:
    """Returns the `GPUInfo.api` value documented for a vendor.

    Raises:
        If the vendor has no documented API string.
    """
    if vendor == Vendor.NVIDIA_GPU:
        return "cuda"
    if vendor == Vendor.AMD_GPU:
        return "hip"
    if vendor == Vendor.APPLE_GPU:
        return "metal"
    raise Error("no documented api for vendor ", vendor)


def test_api_identifies_the_vendor() raises:
    """Every registered target's `api` must name its vendor's API.

    Callers discriminate on the string (`api == "cuda"`, `api == "metal"`), so
    a record carrying the wrong spelling turns those gates silently off instead
    of failing to compile. This pins each record against `_vendor_from_arch`,
    the independent arch-string classifier.
    """
    comptime for i in range(len(_all_targets)):
        comptime arch = rebind[StaticString](_all_targets[i])

        # "cuda" is the generic NVIDIA target: it resolves through runtime GPU
        # detection, so what it names depends on the build's accelerator flag.
        comptime if arch != "cuda":
            comptime vendor = _vendor_from_arch[arch]()

            # An arch the stdlib cannot classify comes from a stdlib plugin,
            # which names its own API for hardware the stdlib has no built-in
            # knowledge of.
            comptime if vendor != Vendor.NO_GPU:
                assert_equal(
                    GPUInfo.from_name[arch]().api,
                    _documented_api(vendor),
                    String("api does not identify the vendor for ", arch),
                )


def test_amd_spellings_resolve() raises:
    """Each accepted spelling of an AMD target must reach the same record.

    The normalization chain rewrites substrings, so a rule meant for one arch
    can corrupt another that merely shares its prefix. That is how `gfx90a` came
    to normalize to the unsupported `gfx90aa`, which left MI250X unreachable
    through every spelling below.
    """
    comptime for i in range(len(_MI250X_SPELLINGS)):
        comptime spelling = rebind[StaticString](_MI250X_SPELLINGS[i])
        assert_equal(
            GPUInfo.from_name[spelling]().name,
            "MI250X",
            String("wrong record for ", spelling),
        )

    assert_equal(GPUInfo.from_name["mi300x"]().name, "MI300X")
    assert_equal(GPUInfo.from_name["amdgpu:gfx942"]().name, "MI300X")
    assert_equal(GPUInfo.from_name["amd:gfx942"]().name, "MI300X")
    assert_equal(GPUInfo.from_name["mi355x"]().name, "MI355X")

    # MI300A shares the gfx942 ISA with MI300X, so this alias is the only way to
    # reach its record: a rule matching "mi300" would silently divert it.
    assert_equal(GPUInfo.from_name["amdgpu:mi300a"]().name, "MI300A")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
