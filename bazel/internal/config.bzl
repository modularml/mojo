"""Private bazel configuration used internally by rules and macros."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@module_versions//:config.bzl", "DEFAULT_PYTHON_VERSION", "DEFAULT_PYTHON_VERSION_UNDERBAR")
load("@with_cfg.bzl//with_cfg/private:select.bzl", "decompose_select_elements")  # buildifier: disable=bzl-visibility
load("//bazel:config.bzl", "DEFAULT_GPU_MEMORY")
load("//bazel/internal:test_resources.bzl", "TEST_RESOURCES")  # buildifier: disable=bzl-visibility

GPU_TEST_ENV = {
    "GPU_ENV_DO_NOT_USE": "$(GPU_CACHE_ENV)",
}

RUNTIME_SANITIZER_DATA = select({
    "@@//:asan_linux_x86_64": ["@clang-linux-x86_64//:lib/clang/22/lib/x86_64-unknown-linux-gnu/libclang_rt.asan.so"],
    "@@//:asan_linux_aarch64": ["@clang-linux-aarch64//:lib/clang/22/lib/aarch64-unknown-linux-gnu/libclang_rt.asan.so"],
    "//conditions:default": [],
}) + select({
    "@@//:asan": ["@@//bazel/internal:lsan-suppressions.txt"],
    "//conditions:default": [],
})

def runtime_sanitizer_env(*, preload = True, location_specifier = "location"):
    env = select({
        "@@//:asan": {
            "LSAN_OPTIONS": "suppressions=$({} @@//bazel/internal:lsan-suppressions.txt)".format(location_specifier),
        },
        "//conditions:default": {},
    })
    if preload:
        env |= select({
            "@@//:asan_linux_x86_64": {
                "LD_PRELOAD": "$({} @clang-linux-x86_64//:lib/clang/22/lib/x86_64-unknown-linux-gnu/libclang_rt.asan.so)".format(location_specifier),
            },
            "@@//:asan_linux_aarch64": {
                "LD_PRELOAD": "$({} @clang-linux-aarch64//:lib/clang/22/lib/aarch64-unknown-linux-gnu/libclang_rt.asan.so)".format(location_specifier),
            },
            "//conditions:default": {},
        })
    return env

def python_version_name(name, python_version):
    if python_version in (DEFAULT_PYTHON_VERSION_UNDERBAR, DEFAULT_PYTHON_VERSION):
        return name
    return "{}_{}".format(name, python_version)

def python_version_tags(python_version):
    tags = ["python-binding-library"]
    if python_version != DEFAULT_PYTHON_VERSION_UNDERBAR:
        tags.extend([
            "no-clang-tidy",
            "no-compile-commands",
            "no-mypy",
        ])
    return tags

def _get_all_constraints(constraints):
    """Extract all possible constraints from the target's 'target_compatible_with'.

    This is complicated because if the 'target_compatible_with' is a select,
    you cannot check if it has a value. This uses an upstream hack to parse the
    select and return all possible values, even if they are not in effect.
    """
    flattened_constraints = []
    for in_select, elements in decompose_select_elements(constraints):
        if type(elements) == type([]):
            flattened_constraints.extend(elements)
        else:
            if in_select and (elements == {} or elements == {"//conditions:default": []}):
                fail("Empty select, delete it")
            flattened_constraints.extend(elements.keys())
            for selected_constraints in elements.values():
                flattened_constraints.extend(selected_constraints)

    return flattened_constraints

def validate_gpu_tags(tags, target_compatible_with):
    """Fail if configured gpu_constraints + tags aren't supported.

    Args:
        tags: The target's 'tags'
        target_compatible_with: The target's 'target_compatible_with'
    """
    if "gpu" in tags:
        return

    has_gpu_constraints = any([
        constraint.endswith(("_gpu", "_gpus"))
        for constraint in _get_all_constraints(target_compatible_with)
    ])
    if has_gpu_constraints:
        fail("tests that have 'gpu_constraints' must specify 'tags = [\"gpu\"],' to be run on CI")

def get_default_exec_properties(tags, target_compatible_with):
    """Return exec_properties that should be shared between different test target types.

    Args:
        tags: The target's 'tags'
        target_compatible_with: The target's 'target_compatible_with'

    Returns:
        A dictionary that should be added to exec_properties of the test target
    """
    gpu_constraints = _get_all_constraints(target_compatible_with)

    exec_properties = {}
    if "requires-network" in tags:
        exec_properties["test.dockerNetwork"] = "bridge"

    if "@//:has_multi_gpu" in gpu_constraints or "//:has_multi_gpu" in gpu_constraints:
        exec_properties["test.resources:gpu-1"] = "0"
        exec_properties["test.resources:gpu-2"] = "0.01"

    if "@//:has_4_gpus" in gpu_constraints or "//:has_4_gpus" in gpu_constraints:
        exec_properties["test.resources:gpu-1"] = "0"
        exec_properties["test.resources:gpu-2"] = "0.01"
        exec_properties["test.resources:gpu-4"] = "0.01"

    return exec_properties

def get_resources_tags(name):
    """Get cpu resource estimates for explicitly listed targets as tags

    Args:
        name: The target's name
    Returns:
        A list of tags for local execution
    """
    resources = TEST_RESOURCES.get("//" + native.package_name() + ":" + name, {})
    if not resources:
        return []
    default_resources = resources.get("default", {})

    # You can't select on tags, so we just return the defaults
    return _format_tags_resources(default_resources)

def _format_tags_resources(resources):
    """Format cpu resource estimates for explicitly listed targets as tags

    Args:
        resources: The target's resources
    Returns:
        A list of tags for local execution
    """
    result = []
    if "cpu" in resources:
        result.append("resources:cpu:{}".format(resources["cpu"]))
    if "memory" in resources:
        result.append("resources:memory:{}".format(resources["memory"]))
    return result

def get_resources_exec_properties(name, test):
    """Get cpu resource estimates for explicitly listed targets

    Args:
        name: The target's name
        test: Whether the target is a test
    Returns:
        A dictionary of exec properties for remote execution
    """
    resources = TEST_RESOURCES.get("//" + native.package_name() + ":" + name, {})
    if not resources:
        return {}
    default_resources = resources.get("default", {})
    return select({
        "@@//:asan": _format_exec_properties_resources(resources.get("asan", {}) if "asan" in resources else default_resources, test, max_cpu = 60, max_memory = 100 * 1024 * 1024 * 1024),
        "@@//:tsan": _format_exec_properties_resources(resources.get("tsan", {}) if "tsan" in resources else default_resources, test, max_cpu = 60, max_memory = 100 * 1024 * 1024 * 1024),
        "@@//:ubsan": _format_exec_properties_resources(resources.get("ubsan", {}) if "ubsan" in resources else default_resources, test, max_cpu = 60, max_memory = 100 * 1024 * 1024 * 1024),
        "@@//:b200_gpu": _format_exec_properties_resources(resources.get("b200", {}) if "b200" in resources else default_resources, test, max_cpu = 32, max_memory = 150 * 1024 * 1024 * 1024),
        "@@//:mi355_gpu": _format_exec_properties_resources(resources.get("mi355", {}) if "mi355" in resources else default_resources, test, max_cpu = 11, max_memory = 150 * 1024 * 1024 * 1024),
        "@platforms//os:macos": _format_exec_properties_resources(resources.get("macos", {}) if "macos" in resources else default_resources, test, max_cpu = 12, max_memory = 30 * 1024 * 1024 * 1024),
        "//conditions:default": _format_exec_properties_resources(default_resources, test, max_cpu = 60, max_memory = 100 * 1024 * 1024 * 1024),
    })

def _format_exec_properties_resources(resources, test, max_cpu = None, max_memory = None):
    """Format resources for the target

    Args:
        resources: The target's resources
        test: Whether the target is a test
        max_cpu: The maximum cpu resource
        max_memory: The maximum memory resource
    Returns:
        A dictionary of exec properties for the target
    """
    result = {
        "debug-disable-measured-task-size": "true",
        "debug-disable-predicted-task-size": "true",
    }

    if test:
        if "cpu" in resources:
            result["test.EstimatedCPU"] = str(min(int(resources["cpu"]), max_cpu))
        if "memory" in resources:
            result["test.EstimatedMemory"] = str(min(int(resources["memory"]), max_memory))
    else:
        if "cpu" in resources:
            result["EstimatedCPU"] = str(min(int(resources["cpu"]), max_cpu))
        if "memory" in resources:
            result["EstimatedMemory"] = str(min(int(resources["memory"]), max_memory))
    return result

def get_default_test_env(exec_properties):
    """Get environment variables that should be shared between different test target types.

    Args:
        exec_properties: The target's 'exec_properties'

    Returns:
        A dictionary that should be added to the test target's 'env'
    """

    # TODO(MOTO-1512): 0.6 accounts for unknown overhead
    gpu_memory_limit = float(exec_properties.get("test.resources:gpu-memory", DEFAULT_GPU_MEMORY))
    adjusted_gpu_memory_limit = gpu_memory_limit - 0.6
    if adjusted_gpu_memory_limit < 0.0:
        fail("GPU memory limit must be at least 1 GiB, got: {}".format(gpu_memory_limit))

    return select({
        "@@//:has_gpu": {
            "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_ONLY": "true",
            "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE": "{}".format(int(adjusted_gpu_memory_limit * 1073741824.0)),
            "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT": "100",
        },
        "//conditions:default": {},
    }) | select({
        # On macOS, the Metal memory manager is disabled on BuildBuddy remote
        # workers (max_cache_size=0), so MEMORY_MANAGER_ONLY must be false to
        # allow fallthrough to direct device allocation.
        "@platforms//os:macos": {
            "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_ONLY": "false",
        },
        "//conditions:default": {},
    }) | select({
        # Sanitizer mode (`--//:gpu_disable_memory_manager`): disable the caching
        # allocator so each `enqueue_create_buffer` is a 1:1 device allocation
        # and compute-sanitizer memcheck/initcheck see true per-buffer bounds.
        # `memory_manager_size=0` + `memory_manager_only=false` routes allocation
        # straight to the device driver (see MemoryManager.cpp onDevice/allocate).
        # Right-biased `|` lets this override the pooled values set above.
        "@@//:gpu_memory_manager_disabled": {
            "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_ONLY": "false",
            "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE": "0",
        },
        "//conditions:default": {},
    })

_TOOLS = {}

def env_for_available_tools(
        *,
        location_specifier = "rootpath",
        os = "unknown"):
    """Get a dictionary of env vars for looking up known tools.

    NOTE: This returns values regardless of if the current dependency tree
    contains the given tools. If a tool is missing that means it should be
    added to data.

    Args:
        location_specifier: The variant of $(location) that we try to emulate with the produced path
        os: Either 'unknown' for macros, or 'linux', or 'macos'

    Returns:
        A dictionary of env vars to be added to a rule that expects to lookup tools
    """

    if location_specifier not in (("execpath", "rootpath")):
        fail("Unsupported location_specifier: {}".format(location_specifier))
    if os not in (("unknown", "linux_aarch64", "linux_x86_64", "macos")):
        fail("Unsupported os: {}".format(os))

    def build_path(label, format_name):
        if location_specifier == "execpath":
            return paths.join("$(BINDIR)", label.workspace_root, label.package, format_name(label.name))
        else:
            return paths.join(label.workspace_root, label.package, format_name(label.name)).replace("external/", "../")

    env = {}
    for label, key in _TOOLS.items():
        env[key] = build_path(label, lambda x: x)

    # Silence buildifier warnings here, since these are accessed from external repos.

    os_specifics = select({
        "@platforms//os:linux": {"LLDB_DEBUGSERVER_PATH": build_path(Label("@llvm-project//lldb:lldb-server"), lambda x: x)},
        "@platforms//os:macos": {"LLDB_DEBUGSERVER_PATH": build_path(Label("@llvm-project//lldb:debugserver"), lambda x: x)},
    }) | select({
        # buildifier: disable=canonical-repository
        "@@//:use_prebuilt_mojo_toolchain_disabled": {},
        # buildifier: disable=canonical-repository
        "@@//:linux_aarch64_prebuilt_mojo_toolchain_enabled": {"MODULAR_MOJO_MAX_PACKAGE_ROOT": build_path(Label("@@+rebuild_wheel+module_platlib_linux_aarch64//:modular"), lambda x: x)},
        # buildifier: disable=canonical-repository
        "@@//:linux_x86_64_prebuilt_mojo_toolchain_enabled": {"MODULAR_MOJO_MAX_PACKAGE_ROOT": build_path(Label("@@+rebuild_wheel+module_platlib_linux_x86_64//:modular"), lambda x: x)},
        # buildifier: disable=canonical-repository
        "@@//:macos_prebuilt_mojo_toolchain_enabled": {"MODULAR_MOJO_MAX_PACKAGE_ROOT": build_path(Label("@@+rebuild_wheel+module_platlib_macos_arm64//:modular"), lambda x: x)},
    })
    if os == "linux_x86_64":
        os_specifics = {
            "LLDB_DEBUGSERVER_PATH": build_path(Label("@llvm-project//lldb:lldb-server"), lambda x: x),
        } | select({
            # buildifier: disable=canonical-repository
            "@@//:use_prebuilt_mojo_toolchain_disabled": {},
            # buildifier: disable=canonical-repository
            "@@//conditions:default": {"MODULAR_MOJO_MAX_PACKAGE_ROOT": build_path(Label("@@+rebuild_wheel+module_platlib_linux_x86_64//:modular"), lambda x: x)},
        })
    elif os == "linux_aarch64":
        os_specifics = {
            "LLDB_DEBUGSERVER_PATH": build_path(Label("@llvm-project//lldb:lldb-server"), lambda x: x),
        } | select({
            # buildifier: disable=canonical-repository
            "@@//:use_prebuilt_mojo_toolchain_disabled": {},
            # buildifier: disable=canonical-repository
            "@@//conditions:default": {"MODULAR_MOJO_MAX_PACKAGE_ROOT": build_path(Label("@@+rebuild_wheel+module_platlib_linux_aarch64//:modular"), lambda x: x)},
        })
    elif os == "macos":
        os_specifics = {
            "LLDB_DEBUGSERVER_PATH": build_path(Label("@llvm-project//lldb:debugserver"), lambda x: x),
        } | select({
            # buildifier: disable=canonical-repository
            "@@//:use_prebuilt_mojo_toolchain_disabled": {},
            # buildifier: disable=canonical-repository
            "@@//conditions:default": {"MODULAR_MOJO_MAX_PACKAGE_ROOT": build_path(Label("@@+rebuild_wheel+module_platlib_macos_arm64//:modular"), lambda x: x)},
        })

    return env | os_specifics
