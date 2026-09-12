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
"""Shared scaffolding for the relay-assisted grouped collectives.

Two groups of GPUs running a grouped collective concurrently use only their
intra-group links; every link between the groups sits idle. Where an even
number of groups lets them pair off, a GPU in the partner group can put those
links to work as a relay, taking on a trailing fraction of the traffic. The
allgather relay is a multicast node and the reduce-scatter relay is a reduce
node, but both need the same gate, the same shard-splitting rule and the same
shape of tuning entry, which live here.
"""

from std.utils import StaticTuple

from internal_utils import TuningConfig

from .device_query import CommTuningConfig
from .sync import MAX_GPUS


@inline(.always)
def _relay_pairs[ngpus: Int, group_size: Int](sm_version: StaticString) -> Bool:
    """Whether adjacent groups pair up to relay for each other at this shape.

    Relaying needs an even number of groups to pair off, both groups of a pair
    to fit the barrier's rank space, and an architecture the recipe has been
    measured on -- the transport is generic but the win turns on how much a
    relay's second hop costs against a direct one. A full-world collective has
    one group, so it is excluded and keeps every link busy anyway.

    Both widths must also be powers of two. Nothing in the transport needs
    that, but every shape this was measured and tested against is one, and a
    ragged pairing is not worth relaying blind: an unvalidated width silently
    takes the plain path instead.

    Parameters:
        ngpus: Total devices in the world.
        group_size: Devices per independent group.

    Args:
        sm_version: The target architecture's version string.
    """
    return (
        sm_version == RELAY_ARCH
        and (ngpus // group_size) % 2 == 0
        and 2 * group_size <= MAX_GPUS
        and group_size.is_power_of_two()
        and ngpus.is_power_of_two()
    )


@inline(.always)
def _relay_slice_vectors[
    ngpus: Int
](num_simd_vectors: Int, relay_percent: Int) -> Int:
    """Vectors of one shard that each of the `ngpus` relays forwards.

    Every rank derives a shard's split from its length alone, so the host and
    all `2 * ngpus` devices agree on it without exchanging anything. Shards are
    split independently, which is what lets the relay path carry ragged groups.

    Parameters:
        ngpus: Number of GPUs in one group, which is also the number of relays
            the trailing part of a shard is spread over.

    Args:
        num_simd_vectors: Whole vectors in the shard; the scalar remainder past
            them belongs to the direct streams.
        relay_percent: Percent of the shard to route through relays.
    """
    return (num_simd_vectors * relay_percent // 100) // ngpus


comptime RELAY_ARCH: StaticString = "CDNA4"
"""The only architecture the relay-assisted allgather is enabled on.

The transport is generic, but the recipe below was measured on MI355X and the
win depends on interconnect specifics -- how much a relay's second hop costs
against a direct one -- that do not carry across vendors. `_allgather_p2p`
gates on this, so every tuning row is implicitly for this arch and none of them
names it.
"""


@fieldwise_init
struct RelayTuningConfig(CommTuningConfig, TrivialRegisterPassable):
    """Tuning-table entry for a relay-assisted grouped collective."""

    var group_size: Int
    """Group width the entry targets, or -1 for the default entry.

    `CommTuningConfig` calls this dimension `ngpus`, but the relay tables are
    keyed on the width of one group rather than on the size of the world, so
    `get_ngpus` reports this field."""

    var num_bytes: Int
    """Largest per-GPU shard, or partition, the entry covers; -1 for the
    default entry."""

    var num_blocks: Int
    """Blocks taking the direct role."""

    var num_relay_blocks: Int
    """Blocks taking the relay role."""

    var relay_percent: Int
    """Percent of every shard, or of every destination's partition, routed
    through the other group. Zero declines the relay path, leaving the plain
    grouped copy."""

    def get_num_blocks(self) -> Int:
        # The launched grid is both roles, and this is what
        # `dispatch_select_comm_config` bound-checks against the barrier's
        # per-block counter arrays -- reporting one role would let a row whose
        # sum exceeds the bound through.
        return self.num_blocks + self.num_relay_blocks

    def get_num_bytes(self) -> Int:
        return self.num_bytes

    def get_sm_version(self) -> StaticString:
        return RELAY_ARCH

    def get_ngpus(self) -> Int:
        return self.group_size

    def write_to(self, mut writer: Some[Writer]):
        """Writes the tuning config as a string.

        Args:
            writer: The writer to write to.
        """
        writer.write(
            self.group_size,
            self.num_bytes,
            self.num_blocks,
            self.num_relay_blocks,
            self.relay_percent,
        )


# Tuning table for `allgather_relay`. `num_bytes` buckets are per-GPU shard
# sizes and must stay in ascending order within an (arch, ngpus) group, since
# the dispatcher takes the first bucket that covers the request.
#
# `relay_percent` is the share of every shard that travels over the idle
# inter-group links. Direct links carry the remaining `1 - f` while relay links
# carry `f`, so the two classes balance -- and the topology floor bottoms out --
# near `f = 0.5`; a relayed byte crosses two links instead of one, which pulls
# the measured optimum just under half.
#
# Sizes above the small-shard cut have no buckets: swept over 2 MB to 48 MB
# shards on MI355X, the best per-size recipe never beat the single default by
# more than 4%, which does not earn a row.
