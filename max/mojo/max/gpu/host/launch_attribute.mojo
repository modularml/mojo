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
"""
GPU Launch Attributes for Kernel Execution Control.

This module provides structures for configuring GPU kernel execution through launch attributes.
It implements a Mojo interface to CUDA's launch attribute system, allowing fine-grained control
over kernel execution characteristics such as memory access policies, synchronization behavior,
cluster dimensions, and resource allocation.

The main components include:
- `LaunchAttributeID`: Identifies different types of launch attributes
- `LaunchAttributeValue`: Stores the value for a specific attribute type
- `LaunchAttribute`: Combines an ID and value to form a complete attribute
- `AccessPolicyWindow`: Configures memory access patterns and caching behavior
- `AccessProperty`: Defines cache persistence properties for memory access

These structures enable optimizing GPU kernel performance by controlling execution parameters
at a granular level, similar to CUDA's native launch attribute system.
"""

from . import Dim
from std.sys import size_of

from std.utils import StaticTuple


@fieldwise_init
struct LaunchAttributeID(Equatable, TrivialRegisterPassable, Writable):
    """Identifies the type of launch attribute for GPU kernel execution.

    This struct represents the various types of launch attributes that can be specified
    when launching GPU kernels or configuring streams and graph nodes. Each attribute
    controls different aspects of kernel execution behavior such as memory access policies,
    synchronization, scheduling, and resource allocation.

    The attributes are compatible with CUDA's launch attribute system and provide
    fine-grained control over kernel execution characteristics.
    """

    var _value: Int32
    """The underlying integer value representing the attribute type."""

    comptime IGNORE = Self(0)
    """Ignored entry, for convenient composition."""

    comptime ACCESS_POLICY_WINDOW = Self(1)
    """Valid for streams, graph nodes, launches."""

    comptime COOPERATIVE = Self(2)
    """Valid for graph nodes, launches."""

    comptime SYNCHRONIZATION_POLICY = Self(3)
    """Valid for streams."""

    comptime CLUSTER_DIMENSION = Self(4)
    """Valid for graph nodes, launches."""

    comptime CLUSTER_SCHEDULING_POLICY_PREFERENCE = Self(5)
    """Valid for graph nodes, launches."""

    comptime PROGRAMMATIC_STREAM_SERIALIZATION = Self(6)
    """Valid for launches. Setting CUlaunchAttributeValue::
    programmaticStreamSerializationAllowed to non-0 signals that the kernel
    will use programmatic means to resolve its stream dependency, so that the
    CUDA runtime should opportunistically allow the grid's execution to overlap
    with the previous kernel in the stream, if that kernel requests the overlap.
    The dependent launches can choose to wait on the dependency using the
    programmatic sync."""

    comptime PROGRAMMATIC_EVENT = Self(7)
    """Valid for launches. Set CUlaunchAttributeValue::programmaticEvent to
    record the event. Event recorded through this launch attribute is guaranteed
    to only trigger after all block in the associated kernel trigger the event.
    A block can trigger the event through PTX launchdep.release or CUDA builtin
    function cudaTriggerProgrammaticLaunchCompletion(). A trigger can also be
    inserted at the beginning of each block's execution if triggerAtBlockStart
    is set to non-0. The dependent launches can choose to wait on the dependency
    using the programmatic sync (cudaGridDependencySynchronize() or equivalent
    PTX instructions). Note that dependents (including the CPU thread calling
    cuEventSynchronize()) are not guaranteed to observe the release precisely
    when it is released. For example, cuEventSynchronize() may only observe
    the event trigger long after the associated kernel has completed. This
    recording type is primarily meant for establishing programmatic dependency
    between device tasks. Note also this type of dependency allows, but does not
    guarantee, concurrent execution of tasks. The event supplied must not be an
    interprocess or interop event. The event must disable timing (i.e. must be
    created with the CU_EVENT_DISABLE_TIMING flag set)."""

    comptime PRIORITY = Self(8)
    """Valid for streams, graph nodes, launches."""

    comptime MEM_SYNC_DOMAIN_MAP = Self(9)
    """Valid for streams, graph nodes, launches."""

    comptime MEM_SYNC_DOMAIN = Self(10)
    """Valid for streams, graph nodes, launches."""

    comptime LAUNCH_COMPLETION_EVENT = Self(12)
    """Valid for launches. Set CUlaunchAttributeValue::launchCompletionEvent to
    record the event. Nominally, the event is triggered once all blocks of the
    kernel have begun execution. Currently this is a best effort. If a kernel
    B has a launch completion dependency on a kernel A, B may wait until A is
    complete. Alternatively, blocks of B may begin before all blocks of A have
    begun, for example if B can claim execution resources unavailable to A
    (e.g. they run on different GPUs) or if B is a higher priority than A.
    Exercise caution if such an ordering inversion could lead to deadlock.
    A launch completion event is nominally similar to a programmatic event
    with triggerAtBlockStart set except that it is not visible to
    cudaGridDependencySynchronize() and can be used with compute capability
    less than 9.0. The event supplied must not be an interprocess or interop
    event. The event must disable timing (i.e. must be created with the
    CU_EVENT_DISABLE_TIMING flag set)."""

    comptime DEVICE_UPDATABLE_KERNEL_NODE = Self(13)
    """Valid for graph nodes, launches. This attribute is graphs-only,
    and passing it to a launch in a non-capturing stream will result in an
    error. CUlaunchAttributeValue::deviceUpdatableKernelNode::deviceUpdatable
    can only be set to 0 or 1. Setting the field to 1 indicates that the
    corresponding kernel node should be device-updatable. On success, a handle
    will be returned via CUlaunchAttributeValue::deviceUpdatableKernelNode::devNode
    which can be passed to the various device-side update functions to update
    the node's kernel parameters from within another kernel. For more
    information on the types of device updates that can be made, as well as the
    relevant limitations thereof, see cudaGraphKernelNodeUpdatesApply. Nodes
    which are device-updatable have additional restrictions compared to regular
    kernel nodes. Firstly, device-updatable nodes cannot be removed from their
    graph via cuGraphDestroyNode. Additionally, once opted-in to this
    functionality, a node cannot opt out, and any attempt to set the
    deviceUpdatable attribute to 0 will result in an error. Device-updatable
    kernel nodes also cannot have their attributes copied to/from another kernel
    node via cuGraphKernelNodeCopyAttributes. Graphs containing one or more
    device-updatable nodes also do not allow multiple instantiation, and neither
    the graph nor its instantiated version can be passed to cuGraphExecUpdate.
    If a graph contains device-updatable nodes and updates those nodes from the
    device from within the graph, the graph must be uploaded with cuGraphUpload
    before it is launched. For such a graph, if host-side executable graph
    updates are made to the device-updatable nodes, the graph must be uploaded
    before it is launched again."""

    comptime PREFERRED_SHARED_MEMORY_CARVEOUT = Self(14)
    """Valid for launches. On devices where the L1 cache and shared memory use
    the same hardware resources, setting CUlaunchAttributeValue::sharedMemCarveout
    to a percentage between 0-100 signals the CUDA driver to set the shared
    memory carveout preference, in percent of the total shared memory for that
    kernel launch. This attribute takes precedence over
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT. This is only a hint,
    and the CUDA driver can choose a different configuration if required for
    the launch."""

    @always_inline("nodebug")
    def __eq__(self, other: Self) -> Bool:
        """Checks if two `LaunchAttribute` instances are equal.

        Compares the underlying integer values of the attributes.

        Args:
            other: The other `LaunchAttribute` instance to compare with.

        Returns:
            True if the attributes are equal, False otherwise.
        """
        return self._value == other._value

    @always_inline("nodebug")
    def __ne__(self, other: Self) -> Bool:
        """Checks if two `LaunchAttribute` instances are not equal.

        Args:
            other: The other `LaunchAttribute` instance to compare with.

        Returns:
            True if the attributes are not equal, False otherwise.
        """
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes the string representation of the attribute to a writer.

        Args:
            writer: The writer to write to.
        """
        return writer.write(self._value)


@fieldwise_init
struct LaunchAttributeValue(Defaultable, TrivialRegisterPassable):
    """Represents a value for a CUDA launch attribute.

    This struct emulates a C union to store different types of launch attribute values.
    It provides a fixed-size storage that can be initialized with different attribute types
    such as AccessPolicyWindow or dimension specifications.

    Note:
        This implementation uses a fixed-size byte array to emulate the union behavior
        defined in the CUDA Driver API's CUlaunchAttributeValue.
    """

    comptime _storage_type = StaticTuple[UInt8, 64]
    var _storage: Self._storage_type
    """Internal storage for the attribute value, represented as a fixed-size byte array."""

    def __init__(out self):
        """Initializes a new `LaunchAttributeValue` with zeroed storage."""
        self._storage = StaticTuple[UInt8, 64](0)

    def __init__(out self, policy: AccessPolicyWindow):
        """Initializes a `LaunchAttributeValue` from an `AccessPolicyWindow`.

        Args:
            policy: The `AccessPolicyWindow` to store in this attribute value.
        """
        var tmp = policy
        var ptr = Pointer(to=tmp)
        self._storage = ptr.unsafe_bitcast[Self._storage_type]()[]

    def __init__(out self, dim: Dim):
        """Initializes a LaunchAttributeValue from a Dim (dimension) object.

        Args:
            dim: The dimension specification to store in this attribute value.
        """
        var tmp = StaticTuple[UInt32, 4](
            UInt32(dim.x()), UInt32(dim.y()), UInt32(dim.z()), 0
        )
        var ptr = Pointer(to=tmp)
        self._storage = ptr.unsafe_bitcast[Self._storage_type]()[]

    def __init__(out self, value: Bool):
        """Initializes a LaunchAttributeValue from a boolean object..

        Args:
            value: The boolean value to store in this attribute value.
        """
        var tmp = StaticTuple[UInt32, 4](UInt32(Int(value)), 0, 0, 0)
        var ptr = Pointer(to=tmp)
        self._storage = ptr.unsafe_bitcast[Self._storage_type]()[]


@fieldwise_init
struct AccessProperty(Equatable, TrivialRegisterPassable, Writable):
    """Specifies performance hint with AccessPolicyWindow for hit_prop and
    miss_prop fields.

    This struct defines cache persistence properties that can be used with
    `AccessPolicyWindow` to control how data is cached during GPU memory accesses.
    It provides hints to the memory subsystem about the expected access patterns,
    which can improve performance for specific workloads.
    """

    var _value: Int32
    """The underlying integer value representing the access property."""

    comptime NORMAL = Self(0)
    """Normal cache persistence with default caching behavior."""

    comptime STREAMING = Self(1)
    """Streaming access is less likely to persist in cache, optimized for single-use data."""

    comptime PERSISTING = Self(2)
    """Persisting access is more likely to persist in cache, optimized for reused data."""

    @always_inline("nodebug")
    def __eq__(self, other: Self) -> Bool:
        """Compares two `AccessProperty` instances for equality.

        Args:
            other: The `AccessProperty` to compare with.

        Returns:
            True if the instances have the same value, False otherwise.
        """
        return self._value == other._value

    @always_inline("nodebug")
    def __ne__(self, other: Self) -> Bool:
        """Compares two `AccessProperty` instances for inequality.

        Args:
            other: The `AccessProperty` to compare with.

        Returns:
            True if the instances have different values, False otherwise.
        """
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of the `AccessProperty` to a writer.

        Args:
            writer: The writer instance to write the formatted string to.
        """
        if self == Self.NORMAL:
            return writer.write("NORMAL")
        if self == Self.STREAMING:
            return writer.write("STREAMING")
        return writer.write("PERSISTING")


@fieldwise_init
struct LaunchAttribute(Defaultable, TrivialRegisterPassable):
    """Represents a complete launch attribute with ID and value.

    This struct combines a `LaunchAttributeID` and `LaunchAttributeValue` to form
    a complete attribute that can be passed to GPU kernel launches. It provides
    a way to specify various execution parameters that control kernel behavior.

    These are CUDA only, and are not supported on any other target.
    """

    var id: LaunchAttributeID
    """The identifier specifying the type of this launch attribute."""

    var __pad: StaticTuple[UInt8, 8 - size_of[LaunchAttributeID]()]
    """Padding to ensure proper alignment of the structure."""

    var value: LaunchAttributeValue
    """The value associated with this launch attribute."""

    def __init__(out self):
        """Initializes a new LaunchAttribute with IGNORE ID and zeroed value."""
        self.id = LaunchAttributeID.IGNORE
        self.__pad = {}
        self.value = {}

    def __init__(out self, id: LaunchAttributeID, value: LaunchAttributeValue):
        """Initializes a `LaunchAttribute` with a specific ID and value.

        Args:
            id: The `LaunchAttributeID` to set.
            value: The `LaunchAttributeValue` to set.
        """
        self.id = id
        self.__pad = {}
        self.value = value

    def __init__(out self, policy: AccessPolicyWindow):
        """Initializes a `LaunchAttribute` from an `AccessPolicyWindow`.

        Creates a launch attribute with `ACCESS_POLICY_WINDOW` ID and the provided policy.

        Args:
            policy: The `AccessPolicyWindow` to use for this attribute.
        """
        self = Self()
        self.id = LaunchAttributeID.ACCESS_POLICY_WINDOW
        self.value = LaunchAttributeValue(policy)

    @staticmethod
    def from_cluster_dim(dim: Dim) -> Self:
        """Creates a `LaunchAttribute` for cluster dimensions.

        Creates a launch attribute with `CLUSTER_DIMENSION` ID and the provided dimensions.

        Args:
            dim: The dimensions to use for this attribute.

        Returns:
            A new `LaunchAttribute` configured with the specified cluster dimensions.
        """
        var res = Self()
        res.id = LaunchAttributeID.CLUSTER_DIMENSION
        res.value = LaunchAttributeValue(dim)
        return res


struct AccessPolicyWindow(
    Defaultable, ImplicitlyCopyable, RegisterPassable, Writable
):
    """Specifies an access policy for a window of memory.

    This struct defines a contiguous extent of memory beginning at base_ptr and
    ending at base_ptr + num_bytes, with associated access policies. It allows
    fine-grained control over how memory is accessed and cached, which can
    significantly impact performance for memory-bound workloads.

    The window is partitioned into segments with different access properties based
    on the hit_ratio. Accesses to "hit segments" use the hit_prop policy, while
    accesses to "miss segments" use the miss_prop policy.

    Note:
        The `num_bytes` value is limited by `CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE`.
        The CUDA driver may align the `base_ptr` and restrict the maximum size.
    """

    var base_ptr: Optional[OpaquePointer[MutUntrackedOrigin]]
    """Starting address of the access policy window. Driver may align it."""

    var num_bytes: Int
    """Size in bytes of the window policy. CUDA driver may restrict the
    maximum size and alignment."""

    var hit_ratio: Float32
    """Specifies percentage of lines assigned hit_prop, rest are assigned
    miss_prop. Value should be between 0.0 and 1.0."""

    var hit_prop: AccessProperty
    """AccessProperty applied to hit segments within the window."""

    var miss_prop: AccessProperty
    """AccessProperty applied to miss segments within the window.
    Must be either NORMAL or STREAMING."""

    def __init__(out self):
        """Initializes a new AccessPolicyWindow with default values."""
        self.base_ptr = None
        self.num_bytes = 0
        self.hit_ratio = 0
        self.hit_prop = AccessProperty.NORMAL
        self.miss_prop = AccessProperty.NORMAL

    def __init__[
        T: AnyType
    ](
        out self,
        *,
        base_ptr: Pointer[T, ...],
        count: Int,
        hit_ratio: Float32,
        hit_prop: AccessProperty = AccessProperty.NORMAL,
        miss_prop: AccessProperty = AccessProperty.NORMAL,
    ):
        """Initializes an `AccessPolicyWindow` for a typed memory region.

        Parameters:
            T: The type of data in the memory region.

        Args:
            base_ptr: Pointer to the start of the memory region.
            count: Number of elements of type T in the memory region.
            hit_ratio: Fraction of the window that should use hit_prop (0.0 to 1.0).
            hit_prop: Access property for hit segments (default: NORMAL).
            miss_prop: Access property for miss segments (default: NORMAL).
        """
        self.base_ptr = Optional[OpaquePointer[MutUntrackedOrigin]](
            base_ptr.unsafe_bitcast[NoneType]()
            .unsafe_mut_cast[True]()
            .unsafe_address_space_cast[.GENERIC]()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )
        self.num_bytes = count * size_of[T]()
        self.hit_ratio = hit_ratio
        self.hit_prop = hit_prop
        self.miss_prop = miss_prop

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of the `AccessPolicyWindow` to a writer.

        This method formats all the fields of the AccessPolicyWindow into a human-readable
        string representation and writes it to the provided writer.

        Args:
            writer: The writer instance to write the formatted string to.
        """
        return writer.write(
            "base_ptr: ",
            self.base_ptr,
            ", num_bytes: ",
            self.num_bytes,
            ", hit_ratio: ",
            self.hit_ratio,
            ", hit_prop: ",
            self.hit_prop,
            ", miss_prop: ",
            self.miss_prop,
        )
