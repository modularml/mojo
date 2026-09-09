# `kprofile`: profile `kbench` output pickle

`kprofile` helps you review and extract insights from `kbench` results stored
in `pkl` files.

You can use `kprofile` with multiple pkl files to display their output one
after another. This groups the output by shape and lets you select different
tuning parameters.

## Review and filter results

- Simply print the top result:

  ```bash
  kprofile output.pkl
  ```

- Find the most frequent values for each parameter in the top 5% of the
  results:

  ```bash
  kprofile output.pkl --top 0.05
  ```

- Print a simplified table that shows the runtime ratio of each entry to the
  top entry:

  ```bash
  kprofile sample.pkl -r
  ```

- Print the 10 best and 10 worst entries:

  ```bash
  kprofile sample.pkl --head 10 --tail 10
  ```

- Group multiple pkl files from different runs and show the two best results
  from each file:

  ```bash
  kprofile file*.pk --head 2
  ```

## Generate checked-in tuning tables

`kprofile` analyzes benchmark results and writes selected configs to YAML. For
a checked-in dispatch table, this YAML file is the *manifest*: the ordered list
of configs that should appear in Mojo.

`tuning_codegen` is the supported path from a manifest to checked-in Mojo. It
renders each config through a snippet by replacing `[@NAME]` placeholders, then
replaces the target file's contents between unique begin and end markers.
Without `--check`, it updates the target file. With `--check`, it reports a diff
and fails if the checked-in region is stale.

Each manifest has a primary snippet. An entry can set `_template` to select a
snippet variant when one table contains configs with different Mojo constructor
shapes. If every entry has the same shape, use only the primary snippet.

The tuning-table workflow is:

1. Run `kprofile` on `kbench` output to create or update the YAML manifest.
2. Review the selected configs.
3. Run `tuning_codegen` to update the marked Mojo region. For example,
   regenerate the SM100 BF16 table:

   ```bash
   ./bazelw run //max/kernels/benchmarks/autotune:tuning_codegen -- \
     --manifest max/kernels/src/linalg/matmul/gpu/sm100_structured/default/tuning_table_sm100_bf16.yaml \
     --snippet max/kernels/src/linalg/matmul/gpu/sm100_structured/default/tuning_sm100.mojo.snippet \
     --target max/kernels/src/linalg/matmul/gpu/sm100_structured/default/tuning_configs.mojo \
     --begin-marker BEGIN-TUNING-LIST-SM100-BF16 \
     --end-marker END-TUNING-LIST-SM100-BF16 \
     --source-label tuning_table_sm100_bf16.yaml
   ```

4. Run `./bazelw run //:format`.
5. Run the table's freshness test. The test regenerates the region in memory
   and fails if the checked-in Mojo differs:

   ```bash
   ./bazelw test \
     //max/kernels/benchmarks/autotune:sm100_bf16_tuning_codegen_test
   ```
