"""One overlay layer: a map of staged relative path -> source File.

Consumers (`mojo_overlay_srcs`, `mojo_merged_precompile`) take an ordered
`layers` list and merge them last-writer-wins per staged path, so any number of
layers can stack. A layer carries the underlying source `File`s (not a generated
tree), so one layer target can feed both the symlink-tree and the
precompile-into-a-dir consumer.

Name staged paths via `srcs` + `strip_prefix` (bulk) and/or `file_map` (per-file
rename); `file_map` wins on a collision.
"""

MojoOverlayLayerInfo = provider(
    doc = "One overlay layer: a dict of staged relative path -> source File.",
    fields = {
        "entries": "dict[str, File]: staged relative path -> source File.",
    },
)

def _rel_from_strip(short_path, prefix):
    if prefix and short_path.startswith(prefix):
        return short_path[len(prefix):]

    # An external-repo filegroup's short_path is `../<canonical>/<rel>`; a fixed
    # startswith prefix can't match it, so locate `prefix` within the path and
    # cut through it.
    if short_path.startswith("../") and prefix:
        idx = short_path.find(prefix)
        if idx >= 0:
            return short_path[idx + len(prefix):]
    return short_path

def _single_file(target):
    files = target[DefaultInfo].files.to_list()
    if len(files) != 1:
        fail("mojo_overlay_layer file_map target %s must provide exactly one file, got %d" % (
            target.label,
            len(files),
        ))
    return files[0]

def _mojo_overlay_layer_impl(ctx):
    entries = {}
    for f in ctx.files.srcs:
        entries[_rel_from_strip(f.short_path, ctx.attr.strip_prefix)] = f

    # `file_map` is applied after `srcs` so an explicit rename shadows a bulk
    # entry at the same staged path.
    for target, rel in ctx.attr.file_map.items():
        entries[rel] = _single_file(target)

    return [
        MojoOverlayLayerInfo(entries = entries),
        DefaultInfo(files = depset(entries.values())),
    ]

mojo_overlay_layer = rule(
    implementation = _mojo_overlay_layer_impl,
    doc = "Declare one overlay layer for `layers` on the mojo staging rules.",
    attrs = {
        "srcs": attr.label_list(
            allow_files = [".mojo"],
            doc = "Sources bulk-staged at their short path minus `strip_prefix`.",
        ),
        "strip_prefix": attr.string(
            doc = "Prefix stripped from each `srcs` short path to get its " +
                  "staged relative path.",
        ),
        "file_map": attr.label_keyed_string_dict(
            allow_files = [".mojo"],
            doc = "Maps an individual source file to the staged relative path " +
                  "it should occupy, shadowing any `srcs` entry at that path.",
        ),
    },
)
