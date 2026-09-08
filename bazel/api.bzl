"""Public API accessors to reduce the number of load statements needed in BUILD.bazel files."""

load("@llvm-project//mlir:tblgen.bzl", _gentbl_cc_library = "gentbl_cc_library", _td_library = "td_library")
load("@rules_pkg//pkg:mappings.bzl", _pkg_filegroup = "pkg_filegroup", _pkg_files = "pkg_files", _strip_prefix = "strip_prefix")
load("@with_cfg.bzl//with_cfg/private:select.bzl", "decompose_select_elements")  # buildifier: disable=bzl-visibility
load("//bazel:mojo_aliases.bzl", "INTERNAL_PACKAGES")
load("//bazel/internal:copy_files.bzl", _copy_files = "copy_files")  # buildifier: disable=bzl-visibility
load("//bazel/internal:dialect_checksum.bzl", _dialect_checksum = "dialect_checksum")  # buildifier: disable=bzl-visibility
load("//bazel/internal:kgen.bzl", _kgen_kernel = "kgen_kernel")  # buildifier: disable=bzl-visibility
load("//bazel/internal:lit.bzl", _lit_tests = "lit_tests")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mef.bzl", "MOJO_DEPS", _mef = "mef")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_cc_binary.bzl", _modular_cc_binary = "modular_cc_binary")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_cc_library.bzl", _modular_cc_library = "modular_cc_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_cc_test.bzl", _modular_cc_test = "modular_cc_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_multi_py_version_test.bzl", _modular_multi_py_version_test = "modular_multi_py_version_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_binary.bzl", _modular_py_binary = "modular_py_binary")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_library.bzl", _modular_py_library = "modular_py_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_test.bzl", _modular_py_test = "modular_py_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_venv.bzl", _modular_py_venv = "modular_py_venv")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_run_binary_test.bzl", _modular_run_binary_test = "modular_run_binary_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_shared_library.bzl", _modular_shared_library = "modular_shared_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_sphinx_docs.bzl", _modular_sphinx_docs = "modular_sphinx_docs")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_versioned_expand_template.bzl", _modular_versioned_expand_template = "modular_versioned_expand_template")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_binary.bzl", _mojo_binary = "mojo_binary")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_filecheck_test.bzl", _mojo_filecheck_test = "mojo_filecheck_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_library.bzl", _mojo_library = "mojo_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_overlay_layer.bzl", _mojo_overlay_layer = "mojo_overlay_layer")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_overlay_srcs.bzl", _mojo_overlay_srcs = "mojo_overlay_srcs")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_shared_library.bzl", _mojo_shared_library = "mojo_shared_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_test.bzl", _mojo_test = "mojo_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_test_environment.bzl", _mojo_test_environment = "mojo_test_environment")  # buildifier: disable=bzl-visibility
load("//bazel/internal:py_repl.bzl", _py_repl = "py_repl")  # buildifier: disable=bzl-visibility
load("//bazel/internal:tablegen.bzl", _driver_option_tablegen = "driver_option_tablegen", _gentbl_mlir_dialect = "gentbl_mlir_dialect", _gentbl_modular_passes = "gentbl_modular_passes")  # buildifier: disable=bzl-visibility
load("//bazel/pip:pip_requirement.bzl", _requirement = "pip_requirement")

dialect_checksum = _dialect_checksum
driver_option_tablegen = _driver_option_tablegen
gentbl_mlir_dialect = _gentbl_mlir_dialect
gentbl_cc_library = _gentbl_cc_library
gentbl_modular_passes = _gentbl_modular_passes
lit_tests = _lit_tests
kgen_kernel = _kgen_kernel
modular_multi_py_version_test = _modular_multi_py_version_test
modular_py_library = _modular_py_library
modular_py_venv = _modular_py_venv
modular_run_binary_test = _modular_run_binary_test
modular_versioned_expand_template = _modular_versioned_expand_template
mojo_overlay_layer = _mojo_overlay_layer
mojo_overlay_srcs = _mojo_overlay_srcs
mojo_test = _mojo_test
mojo_filecheck_test = _mojo_filecheck_test
modular_sphinx_docs = _modular_sphinx_docs
mojo_test_environment = _mojo_test_environment
pkg_files = _pkg_files
pkg_filegroup = _pkg_filegroup
py_repl = _py_repl
requirement = _requirement
strip_prefix = _strip_prefix
td_library = _td_library

_OVERRIDE_DEFINES = {
    "MOJO_COMPILER_ACCELERATOR_SUPPORT": "0",
    "MODULAR_KGEN_PROFILING_ENABLED": "0",
    "MLRT_ACCELERATOR_SUPPORT": "0",
}

def modular_py_test(tags = [], **kwargs):
    if "external-exclusive" in tags:
        tags.append("exclusive")
    _modular_py_test(tags = tags, **kwargs)

def _process_define_list(defines):
    # poor dev's type check, for sanity:
    if type(defines) != type([]):
        fail("_process_define_list: expected list")
    result = []
    for define in defines:
        key = define.split("=")[0]
        override = _OVERRIDE_DEFINES.get(key)
        if override:
            result.append(key + "=" + override)
        else:
            result.append(define)
    return result

def _process_defines(defines):
    if type(defines) == type([]):
        return _process_define_list(defines)
    else:
        # Decompose the select()s, process, and recombine
        result = []
        for is_select, elements in decompose_select_elements(defines):
            if is_select:
                new_select = {}
                for key, values in elements.items():
                    new_select[key] = _process_define_list(values)
                result += select(new_select)
            else:
                result += _process_define_list(elements)
        return result

def _process_cc_deps(data, deps):
    # TODO: This will break in the presence of select()s
    new_deps = []
    needs_wheel = False
    for dep in deps:
        if dep == "//max/internal:max":
            new_deps.append("@modular_wheel//:max_lib")
            needs_wheel = True
        else:
            new_deps.append(dep)

    return {
        "deps": new_deps,
        "data": data + (["@modular_wheel//:wheel"] if needs_wheel else []),
    }

def _process_mojo_deps(deps):
    # TODO: This will break in the presence of select()s
    new_deps = []
    imports_max = False
    needs_compiler_rt = False
    for dep in deps:
        if dep in INTERNAL_PACKAGES:
            new_deps.append("@modular_wheel//:" + dep.split("/")[-1])
            imports_max = True
        elif dep == "//MLRT:Driver/DeviceContext":
            new_deps.append("@modular_wheel//:AsyncRTMojoBindings_lib")
        elif dep == "//Mojo:CompilerRT":
            needs_compiler_rt = True
        else:
            new_deps.append(dep)

    if imports_max and "//max:max_mojo" not in new_deps:
        new_deps.append("//max:max_mojo")

    if needs_compiler_rt:
        new_deps += select({
            "//:use_prebuilt_mojo_toolchain_disabled": ["//Mojo:CompilerRT"],
            "//conditions:default": ["@modular_wheel//:CompilerRT_lib"],
        })

    return new_deps

# Ignore internal_deps for public builds
# buildifier: disable=unused-variable
def modular_cc_binary(data = [], deps = [], internal_deps = [], defines = [], local_defines = [], **kwargs):
    _modular_cc_binary(
        local_defines = _process_defines(local_defines),
        defines = _process_defines(defines),
        **(kwargs | _process_cc_deps(
            data = data,
            deps = deps,
        ))
    )

# Ignore internal_deps for public builds
# buildifier: disable=unused-variable
def modular_cc_library(name, data = [], deps = [], internal_deps = [], defines = [], local_defines = [], **kwargs):
    if name in ["Profiling", "ProfilerHostGlue"]:
        # Provide TimeProfiler for now since that may be what they're actually after
        _modular_cc_library(name = name, deps = ["//Support:TimeProfiler"])
        return

    _modular_cc_library(
        name = name,
        local_defines = _process_defines(local_defines),
        defines = _process_defines(defines),
        **(kwargs | _process_cc_deps(
            data = data,
            deps = deps,
        ))
    )

# Ignore internal_deps for public builds
# buildifier: disable=unused-variable
def modular_cc_test(data = [], deps = [], internal_deps = [], defines = [], local_defines = [], **kwargs):
    _modular_cc_test(
        local_defines = _process_defines(local_defines),
        defines = _process_defines(defines),
        **(kwargs | _process_cc_deps(
            data = data,
            deps = deps,
        ))
    )

# Ignore internal_deps for public builds
# buildifier: disable=unused-variable
def modular_shared_library(name = None, internal_deps = [], defines = [], local_defines = [], **kwargs):
    if name in ["MAXProfilerPlugin"]:
        # Provide TimeProfiler for now since that may be what they're actually after
        _modular_cc_library(name = name, deps = ["//Support:TimeProfiler"])
        return

    _modular_shared_library(
        name = name,
        local_defines = _process_defines(local_defines),
        defines = _process_defines(defines),
        **kwargs
    )

def modular_generate_stubfiles(name, pyi_srcs, deps = [], tags = [], **_kwargs):
    modular_py_library(
        name = name,
        pyi_srcs = pyi_srcs,
        deps = deps + ["@modular_wheel//:wheel"],
        tags = tags + ["no-pydeps"],  # Pydeps works internally but not externally
    )

# Ignore use_production_compiler_for_asan for public builds
# buildifier: disable=unused-variable
def mojo_library(deps = [], use_production_compiler_for_asan = None, **kwargs):
    _mojo_library(
        deps = _process_mojo_deps(deps),
        **kwargs
    )

# Ignore use_production_compiler_for_asan for public builds
# buildifier: disable=unused-variable
def mojo_shared_library(deps = [], use_production_compiler_for_asan = None, **kwargs):
    _mojo_shared_library(
        deps = _process_mojo_deps(deps),
        **kwargs
    )

def mojo_binary(deps = [], **kwargs):
    _mojo_binary(
        deps = _process_mojo_deps(deps),
        **kwargs
    )

def modular_py_binary(mojo_deps = [], **kwargs):
    _modular_py_binary(mojo_deps = _process_mojo_deps(mojo_deps), **kwargs)

# buildifier: disable=function-docstring
def mef(**kwargs):
    _mef(mojo_deps = _process_mojo_deps(MOJO_DEPS), **kwargs)

# buildifier: disable=function-docstring
def copy_files(srcs, **kwargs):
    new_srcs = []
    for src in srcs:
        if src.startswith("//GraphCompiler:"):
            if "@modular_wheel//:tblgen_python_srcs" not in new_srcs:
                new_srcs.append("@modular_wheel//:tblgen_python_srcs")
        else:
            new_srcs.append(src)

    _copy_files(srcs = new_srcs, **kwargs)

def _noop(**_kwargs):
    pass

install_docs = _noop
mlir_nanobind = _noop
modular_nanobind_extension = _noop
modular_nanobind_library = _noop
modular_python_binding_library_test = _noop
