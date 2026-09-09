"""Module extension to configure LLVM"""

load("@llvm-raw//utils/bazel:configure.bzl", _llvm_configure = "llvm_configure")

BACKENDS = [
    "AArch64",
    "RISCV",
    "X86",
]

def _llvm_project_impl(module_ctx):
    targets = {t: None for t in BACKENDS}
    for mod in module_ctx.modules:
        for tag in mod.tags.configure:
            for t in tag.extra_targets:
                targets[t] = None

    _llvm_configure(
        name = "llvm-project",
        targets = sorted(targets.keys()),
    )

    return module_ctx.extension_metadata(reproducible = True)

_configure = tag_class(
    attrs = {
        "extra_targets": attr.string_list(
            doc = "Additional LLVM backends to configure alongside default backends.",
        ),
    },
)

# NOTE: exported as `llvm_configure` (not `llvm_project`) on purpose: the
# canonical bzlmod repo name is derived from the extension symbol, so this keeps
# it `@@+llvm_configure+llvm-project`. Renaming
# the symbol would rename the repo and break those references.
llvm_configure = module_extension(
    implementation = _llvm_project_impl,
    tag_classes = {"configure": _configure},
    doc = "Configures LLVM as `@llvm-project` with the selected backends.",
)
