"""Stage an ordered stack of overlay `layers` into one Mojo source tree.

Produces staged `.mojo` files (not a compiled package) so a `mojo_library` can
consume them as `srcs`; all share one staging root so the compiler resolves
module paths relative to it. `layers` merge last-writer-wins per staged path;
`stage_root` is prepended so the tree roots at the package name (e.g. `std/`);
`exclude` drops orphaned paths. This lets a build flag swap individual stdlib
files without touching the checked-in tree.
"""

load(":mojo_overlay_layer.bzl", "MojoOverlayLayerInfo")

def _mojo_overlay_srcs_impl(ctx):
    # Merge layers in order (later shadows earlier), dropping excluded paths.
    exclude = {p: True for p in ctx.attr.exclude}
    staged = {}
    for layer in ctx.attr.layers:
        for rel, f in layer[MojoOverlayLayerInfo].entries.items():
            if rel in exclude:
                continue
            staged[rel] = f

    # Stage in sorted-path order so the package root `__init__.mojo` (which sorts
    # first) is the consuming library's `srcs[0]`, fixing the compiler's root.
    root = ctx.attr.stage_root
    outs = []
    for rel in sorted(staged.keys()):
        out = ctx.actions.declare_file(ctx.label.name + "/" + root + rel)
        ctx.actions.symlink(output = out, target_file = staged[rel])
        outs.append(out)

    return [DefaultInfo(files = depset(outs))]

mojo_overlay_srcs = rule(
    implementation = _mojo_overlay_srcs_impl,
    doc = "Stage an ordered stack of `mojo_overlay_layer`s into one source tree.",
    attrs = {
        "layers": attr.label_list(
            providers = [MojoOverlayLayerInfo],
            doc = "Ordered overlay layers; a later layer shadows an earlier " +
                  "one where their staged paths collide.",
        ),
        "stage_root": attr.string(
            doc = "Prepended to every staged path so the tree roots at the " +
                  "package name (for example `std/`). Defaults to no prefix.",
        ),
        "exclude": attr.string_list(
            doc = "Staged relative paths to drop entirely, for example backend " +
                  "modules orphaned once an overlay replaces their only importer.",
        ),
    },
)
