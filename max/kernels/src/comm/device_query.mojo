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
"""Provides device query utilities for communication primitives."""

from std.sys.info import _accelerator_arch
from internal_utils import TuningConfig, Table
from max.gpu.host.info import GPUInfo

comptime KB = 1 << 10
comptime MB = 1 << 20
comptime GB = 1 << 30


trait CommTuningConfig(TuningConfig):
    """Tuning-table entry for a multi-GPU communication collective.

    Extends `TuningConfig` with the four dimensions that drive kernel-launch
    selection: SM version, GPU count, total data size, and thread-block count.
    Implement this trait to supply custom tuning tables to
    `dispatch_select_comm_config`.
    """

    def get_num_blocks(self) -> Int:
        """Returns the number of thread blocks to launch for this configuration.

        Returns:
            The thread-block count, which must not exceed 512
            (`MAX_NUM_BLOCKS_UPPER_BOUND`).
        """
        ...

    def get_num_bytes(self) -> Int:
        """Returns the maximum input size in bytes covered by this entry.

        `dispatch_select_comm_config` selects the first entry whose
        `get_num_bytes()` is at least the actual transfer size.

        Returns:
            The upper-bound byte count for this tuning entry, or -1 for the
            default (catch-all) entry.
        """
        ...

    def get_sm_version(self) -> StaticString:
        """Returns the SM version string this entry targets.

        Returns:
            A string such as `"sm_90a"` or `"sm_100a"`, or `"default"` for
            the architecture-agnostic fallback entry.
        """
        ...

    def get_ngpus(self) -> Int:
        """Returns the GPU count this entry targets.

        Returns:
            The number of participating GPUs, or -1 for the default
            (catch-all) entry.
        """
        ...


@fieldwise_init
struct DefaultCommTuningConfig(CommTuningConfig, TrivialRegisterPassable):
    """
    Parameters:
        ngpus: Number of GPUs for running allreduce.
        num_bytes: Total number of input bytes supported by the config.
        sm_version: SM version (as string).
        num_blocks: Number of thread blocks for running allreduce.
    """

    var ngpus: Int
    var num_bytes: Int
    var sm_version: StaticString
    var num_blocks: Int

    def get_num_blocks(self) -> Int:
        return self.num_blocks

    def get_num_bytes(self) -> Int:
        return self.num_bytes

    def get_sm_version(self) -> StaticString:
        return self.sm_version

    def get_ngpus(self) -> Int:
        return self.ngpus

    def write_to(self, mut writer: Some[Writer]):
        """Writes the tuning config as a string.

        Args:
            writer: The writer to write to.
        """
        writer.write(
            self.ngpus, self.num_bytes, self.sm_version, self.num_blocks
        )


@always_inline
def dispatch_select_comm_config[
    TuningTableType: CommTuningConfig,
    //,
    ngpus: Int,
    sm_version: StaticString,
    tuning_table: Table[TuningTableType],
](num_bytes: Int) -> TuningTableType:
    """
    This function searches for tuning configs with matching sm_version
    and ngpus. If such configs are found, then the search continues for
    finding the config x where num_bytes <= x.num_bytes.

    Falls back to the arch-specific default (ngpus=-1, num_bytes=-1,
    matching sm_version), or if none exists, to the global default
    (ngpus=-1, num_bytes=-1, sm_version="default").

    Parameters:
        TuningTableType: The tuning-config entry type, constrained to
            `CommTuningConfig`.
        ngpus: Number of participating GPUs to select a config for.
        sm_version: Target SM version string to match, such as `"sm_90a"`.
        tuning_table: Compile-time table of tuning configs to search.

    Args:
        num_bytes: Actual transfer size in bytes to select a config for.
    """

    # Validate that every entry has num_blocks <= 512 (MAX_NUM_BLOCKS_UPPER_BOUND
    # from sync.mojo). _multi_gpu_barrier indexes Signal.self_counter and
    # Signal.peer_counter with block_idx.x; those arrays are statically sized
    # to MAX_NUM_BLOCKS_UPPER_BOUND, so an entry exceeding 512 would silently
    # corrupt barrier state.
    def _entry_exceeds_block_bound(x: tuning_table.type) {} -> Bool:
        return x.get_num_blocks() > 512

    comptime _over_limit = tuning_table.query_index(
        rule=_entry_exceeds_block_bound
    )
    comptime assert (
        len(_over_limit) == 0
    ), "tuning_table entry has num_blocks > MAX_NUM_BLOCKS_UPPER_BOUND (512)"

    # get default entry: prefer arch-specific, fall back to global default
    def rule_eq_arch_default(x: tuning_table.type) {} -> Bool:
        return (
            x.get_ngpus() == -1
            and x.get_num_bytes() == -1
            and x.get_sm_version() == sm_version
        )

    def rule_eq_global_default(x: tuning_table.type) {} -> Bool:
        return (
            x.get_ngpus() == -1
            and x.get_num_bytes() == -1
            and x.get_sm_version() == "default"
        )

    comptime arch_default_idx = tuning_table.query_index(
        rule=rule_eq_arch_default
    )
    comptime global_default_idx = tuning_table.query_index(
        rule=rule_eq_global_default
    )
    comptime default_idx = arch_default_idx if len(
        arch_default_idx
    ) > 0 else global_default_idx
    comptime assert len(default_idx) > 0, (
        "tuning_table must have a default entry for sm_version: "
        + sm_version
        + " or a global default entry (sm_version='default')"
    )
    comptime default_entry = tuning_table.configs[default_idx[0]]

    # narrowing the search space to matching sm_version and ngpus
    def rule_eq_arch_ngpus(x: tuning_table.type) {} -> Bool:
        return x.get_sm_version() == sm_version and x.get_ngpus() == ngpus

    comptime search_domain = tuning_table.query_index(rule=rule_eq_arch_ngpus)

    comptime if not search_domain:
        return default_entry

    # get all static num_bytes values in table within the search space
    comptime all_num_bytes_values = tuning_table.query_values[
        Int, domain=search_domain
    ](rule=lambda (x: tuning_table.type) -> Int: x.get_num_bytes())

    comptime for nb in all_num_bytes_values:

        def rule_eq_nb(x: tuning_table.type) {} -> Bool:
            return x.get_num_bytes() == nb

        # Find the fist config x with input 'num_bytes <= x.num_bytes'
        if num_bytes <= nb:
            comptime idx_list = tuning_table.query_index[domain=search_domain](
                rule=rule_eq_nb
            )

            comptime if idx_list:
                comptime entry = tuning_table.configs[idx_list[0]]
                return entry
            else:
                break

    return default_entry


def get_sm_version() -> StaticString:
    """Returns the SM version string for the current compile target.

    Queries the GPU info for the default accelerator architecture and returns
    its version identifier (e.g. `"sm_90a"` for Hopper, `"sm_100a"` for
    Blackwell). The result is a comptime constant derived from
    `GPUInfo.from_name`.

    Returns:
        The SM version string for the target GPU architecture.
    """
    comptime default_device_info = GPUInfo.from_name[_accelerator_arch()]()
    return default_device_info.version
