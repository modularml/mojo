"""A rule for creating shared libraries based on our common patterns."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//bazel:config.bzl", "ALLOW_UNUSED_TAG")
load("//tools/build_defs/cc:link_hack.bzl", "link_hack")  # See link_hack.bzl for details
load(":modular_clang_tidy_test.bzl", "modular_clang_tidy_test")
load(":modular_strip_binary.bzl", "register_strip_action")

def _shared_library_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    all_deps = ctx.attr.deps + ([ctx.attr._link_extra_lib] if ctx.attr._link_extra_lib else [])
    deps_cc_infos = [dep[CcInfo] for dep in all_deps]
    deps_linking_contexts = [cc_info.linking_context for cc_info in deps_cc_infos]
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    expanded_copts = []
    for copt in ctx.attr.copts:
        expanded_copts.append(ctx.expand_make_variables("copts", copt, {}))

    # Assumes the default repository layout; rules_cc's equivalent handles
    # --experimental_sibling_repository_layout, but the accessor for it is
    # private API.
    # https://github.com/bazelbuild/rules_cc/blob/99a85777cfdb897e3ea2de51ecd78774f6cfadab/cc/common/cc_helper.bzl#L804-L833
    if ctx.label.workspace_name:
        package_path = paths.join("external", ctx.label.workspace_name, ctx.label.package)
    else:
        package_path = ctx.label.package

    includes = []
    for include in ctx.attr.includes:
        include_path = paths.join(package_path, include)
        includes.append(include_path)

        # Generated headers (tablegen output, for example) live under the
        # output tree, so each include dir needs its output-tree twin too.
        if ctx.genfiles_dir.path != ctx.bin_dir.path:
            includes.append(paths.join(ctx.genfiles_dir.path, include_path))
        includes.append(paths.join(ctx.bin_dir.path, include_path))

    expanded_local_defines = []
    for opt in ctx.attr.local_defines:
        expanded_local_defines.append(
            ctx.expand_location(
                opt,
                targets = ctx.attr.additional_compiler_inputs,
            ),
        )

    compilation_context, compilation_outputs = cc_common.compile(
        actions = ctx.actions,
        cc_toolchain = cc_toolchain,
        compilation_contexts = [cc_info.compilation_context for cc_info in deps_cc_infos],
        feature_configuration = feature_configuration,
        local_defines = expanded_local_defines,
        defines = ctx.attr.defines,
        name = ctx.label.name,
        private_hdrs = [x for x in ctx.files.srcs if x.basename.endswith((".h", ".hpp", ".inc"))],
        public_hdrs = ctx.files.hdrs,
        textual_hdrs = ctx.files.textual_hdrs,
        srcs = [x for x in ctx.files.srcs if x.basename.endswith((".c", ".cpp"))],
        includes = includes,
        user_compile_flags = expanded_copts,
        additional_inputs = ctx.files.additional_compiler_inputs,
    )

    linkopts = []
    for opt in ctx.attr.linkopts:
        linkopts.append(
            ctx.expand_location(
                opt,
                targets = ctx.attr.additional_linker_inputs,
            ),
        )

    # Only set if name is not using the default logic
    link_kwargs = {}
    output_name = ctx.attr.output_name or ctx.label.name
    dsym_name = "lib{}.dylib.dSYM".format(output_name)
    if ctx.attr.shared_lib_name:
        if ctx.attr.output_name:
            fail("Cannot set both 'shared_lib_name' and 'output_name'.")

        shared_lib_name = ctx.expand_make_variables(
            "shared_lib_name",
            ctx.attr.shared_lib_name,
            {},
        )
        dsym_name = shared_lib_name + ".dSYM"
        link_kwargs["main_output"] = ctx.actions.declare_file(shared_lib_name)

    additional_linker_outputs = []
    link_variables = {}
    output_group_kwargs = {}
    dwarf_outputs = []
    if cc_common.is_enabled(feature_configuration = feature_configuration, feature_name = "generate_dsym_file"):
        dsym_file = ctx.actions.declare_directory(dsym_name)
        dwarf_outputs.append(dsym_file)
        link_variables["dsym_path"] = dsym_file.path
        additional_linker_outputs.append(dsym_file)

    linking_outputs = link_hack(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        compilation_outputs = compilation_outputs,
        linking_contexts = deps_linking_contexts,
        emit_interface_shared_library = True,
        name = output_name,
        output_type = "dynamic_library",
        user_link_flags = linkopts,
        additional_inputs = ctx.files.additional_linker_inputs,
        additional_outputs = additional_linker_outputs,
        variables_extension = link_variables,
        **link_kwargs
    )

    dwarf_outputs.append(linking_outputs.library_to_link.resolved_symlink_dynamic_library)
    output_group_kwargs["modular_dwarf"] = depset(dwarf_outputs)

    # TODO: Change name to output beside the main library once we can rename in release packaging
    stripped_output = ctx.actions.declare_file(
        ctx.label.name + ".stripped/" + linking_outputs.library_to_link.resolved_symlink_dynamic_library.basename,
    )
    register_strip_action(
        ctx = ctx,
        input_file = linking_outputs.library_to_link.resolved_symlink_dynamic_library,
        output_file = stripped_output,
    )

    transitive_runfiles = []
    transitive_libraries = []
    for target in all_deps:
        transitive_runfiles.append(target[DefaultInfo].default_runfiles)

    for linking_context in deps_linking_contexts:
        for linker_input in linking_context.linker_inputs.to_list():
            for library in linker_input.libraries:
                if library.dynamic_library and not library.pic_static_library and not library.static_library:
                    transitive_libraries.append(depset([library]))
                    transitive_runfiles.append(ctx.runfiles(transitive_files = depset([library.dynamic_library])))

    return [
        coverage_common.instrumented_files_info(
            ctx,
            dependency_attributes = ["deps"],
            source_attributes = ["srcs"],
        ),
        OutputGroupInfo(
            modular_stripped = depset([stripped_output]),
            **output_group_kwargs
        ),
        DefaultInfo(
            executable = linking_outputs.library_to_link.resolved_symlink_dynamic_library,
            runfiles = ctx.runfiles().merge_all(transitive_runfiles),
        ),
        CcInfo(
            compilation_context = compilation_context,
            linking_context = cc_common.create_linking_context(
                linker_inputs = depset([
                    cc_common.create_linker_input(
                        owner = ctx.label,
                        libraries = depset(
                            [linking_outputs.library_to_link],
                            transitive = transitive_libraries,
                        ),
                    ),
                ]),
            ),
        ),
    ]

_shared_library = rule(
    implementation = _shared_library_impl,
    attrs = {
        "copts": attr.string_list(),
        "deps": attr.label_list(providers = [CcInfo]),
        "hdrs": attr.label_list(allow_files = True),
        "textual_hdrs": attr.label_list(allow_files = True),
        "additional_linker_inputs": attr.label_list(allow_files = True),
        "additional_compiler_inputs": attr.label_list(allow_files = True),
        "linkopts": attr.string_list(),
        "defines": attr.string_list(),
        "local_defines": attr.string_list(),
        "shared_lib_name": attr.string(),
        "srcs": attr.label_list(allow_files = [".c", ".cpp", ".h", ".hpp", ".inc"]),
        "includes": attr.string_list(),
        "output_name": attr.string(),
        "_linux_constraint": attr.label(default = Label("@platforms//os:linux")),
        "_link_extra_lib": attr.label(
            default = "@bazel_tools//tools/cpp:link_extra_lib",
            providers = [CcInfo],
        ),
    },
    provides = [CcInfo],
    toolchains = use_cpp_toolchain(),
    fragments = ["cpp"],
)

def modular_shared_library(
        name,
        srcs,
        copts = [],
        deps = [],
        internal_deps = [],
        hdrs = [],
        textual_hdrs = [],
        defines = [],
        local_defines = [],
        additional_compiler_inputs = [],
        additional_linker_inputs = [],
        linkopts = [],
        includes = ["include"],
        visibility = None,
        output_name = None,
        shared_lib_name = None,
        target_compatible_with = None,
        tags = [],
        testonly = False,
        toolchains = [],
        features = []):
    """Create a shared library target

    NOTE: This rule creates a few underlying targets that can be used in
    different situations. Ideally the shared library is the primary target
    being used, but you can also use the headers only target, as well as the
    statically linkable target if you know there is no risk of duplicating the
    contents in the same process.

    Args:
        name: The name of the final shared library target
        srcs: C++ source files to include in the target
        deps: Deps required to compile the target
        internal_deps: Same as `deps`, but excluded for external builds.
        hdrs: Headers to propagate for users of the shared library
        textual_hdrs: Headers that are textually included and cannot be built on their own
        additional_compiler_inputs: Non-source files that must be available when compiling the library
        additional_linker_inputs: Passed through to the shared library
        linkopts: Passed through to the shared library
        defines: Passed through to cc_library
        local_defines: Passed through to cc_library
        copts: Passed through to cc_library
        includes: Include search paths to propagate to dependents
        visibility: See cc_library docs
        output_name: The name of the library on disk, if omitted 'name' is used
        shared_lib_name: The full name of the shared library on disk, including prefix and file. extension.
        target_compatible_with: Passed through
        tags: Tags set on all underlying targets
        testonly: If true, only used in tests
        toolchains: Toolchains to use for producing the shared library
        features: See upstream docs
    """
    _shared_library(
        name = name,
        copts = copts,
        deps = deps + internal_deps,
        hdrs = hdrs,
        textual_hdrs = textual_hdrs,
        srcs = srcs,
        includes = includes,
        visibility = visibility,
        output_name = output_name,
        shared_lib_name = shared_lib_name,
        additional_compiler_inputs = additional_compiler_inputs,
        additional_linker_inputs = additional_linker_inputs,
        linkopts = linkopts,
        defines = defines,
        local_defines = local_defines + [
            "MODULAR_BUILDING_LIBRARY",
        ],
        target_compatible_with = target_compatible_with,
        tags = tags,
        testonly = testonly,
        toolchains = toolchains,
        features = features,
    )

    native.filegroup(
        name = name + ".stripped",
        output_group = "modular_stripped",
        srcs = [":" + name],
        tags = ["manual", ALLOW_UNUSED_TAG],
        testonly = testonly,
        visibility = visibility,
    )

    native.filegroup(
        name = name + ".dwarf",
        output_group = "modular_dwarf",
        srcs = [":" + name],
        tags = ["manual", ALLOW_UNUSED_TAG],
        testonly = testonly,
        visibility = visibility,
    )

    modular_clang_tidy_test(
        name = name,
        hdrs = hdrs,
        srcs = srcs,
        copts = copts,
        tags = tags,
        additional_compiler_inputs = additional_compiler_inputs,
    )
