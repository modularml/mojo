"""Generate MEF files from python Graphs."""

load("@cfg_workaround.bzl", "TARGET_CONSTRAINTS")
load(":modular_genrule.bzl", "modular_genrule")
load(":modular_py_binary.bzl", "modular_py_binary")

# All transitive mojo dependencies of //max:kernels
MOJO_DEPS = [
    # Internal-only kernel packages. Each is imported by a graph-op
    # registration in //max:builtin_kernels, so its .mojopkg has to be on the
    # graph compiler's runtime import path -- otherwise the whole
    # builtin_kernels package body fails to resolve. This list is flat, not
    # transitive, so adding a //Kernels/... dep to //max:builtin_kernels
    # requires adding it here too. The open-source build has none of these
    # packages; the OSS `mef` wrapper in //oss/modular/bazel:api.bzl rewrites
    # them to their prebuilt @modular_wheel equivalents.
    "//Kernels/lib/attn_res",
    "//Kernels/lib/kda",
    "//Kernels/lib/matmul_rs",
    "//Kernels/lib/msa",
    "//Kernels/src/mega_ffn",
    "//max:algorithm",
    "//max:builtin_kernels",
    "//max:builtin_primitives",
    "//max:_cublas",
    "//max:_cudnn",
    "//max:_cufft",
    "//max:_miopen",
    "//max:_rocblas",
    "//max:comm",
    "//max:extensibility",
    "//max:internal_utils",
    "//max:kv_cache",
    "//max:layout",
    "//max:linalg",
    "//max:nn",
    "//max:pipeline",
    "//max:quantization",
    "//max:shmem",
    "//max:state_space",
    "//max:structured_kernels",
    "//max:weights_registry",
    "//max:max_mojo",
    "@mojo//:std",
]

def mef(name, src, args = [], target_compatible_with = [], mojo_deps = MOJO_DEPS, **kwargs):
    """Generate mef file from a python executable.

    The generated file will be {name}.mef.

    Args:
        name: The target name for the generated file.
        src: .py file generating the MEF file.
        args: Args added to the python file's execution
        target_compatible_with: Constraints for platform execution
        mojo_deps: Additional mojo dependencies to add to the python file's execution
        **kwargs: forwarded to the `py_binary` target
    """
    py_binary_name = name + ".py_binary"
    mef_name = name + ".mef"

    modular_py_binary(
        name = py_binary_name,
        srcs = [src],
        main = src,
        target_compatible_with = target_compatible_with,
        mojo_deps = mojo_deps,
        **kwargs
    )

    modular_genrule(
        name = name,
        outs = [mef_name],
        exec_compatible_with = TARGET_CONSTRAINTS + target_compatible_with,
        target_compatible_with = target_compatible_with + select({
            "//:asan": ["@platforms//:incompatible"],  # TODO: SDLC-4011
            "//conditions:default": [],
        }),
        cmd = " ".join([
            "MODULAR_HOME=.",
            "MODULAR_MOJO_MAX_IMPORT_PATH=" + ",".join([
                "$$(dirname $(location {}))".format(dep)
                for dep in mojo_deps
            ]),
            "$(location :" + py_binary_name + ")",
            "$(location :" + mef_name + ")",
        ] + args),
        tools = [":" + py_binary_name] + mojo_deps,
    )
