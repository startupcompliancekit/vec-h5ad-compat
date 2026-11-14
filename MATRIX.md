# Measured reader matrix

How the lookup table in `h5ad_compat.py` was obtained. Every file is a tiny
synthetic AnnData (6 cells x 4 genes of random numbers) written by
`python matrix.py write DIR` and opened by `python matrix.py read DIR`, which
calls `anndata.read_h5ad`. Python 3.11, numpy 2.4.6, h5py 3.16.0, scipy 1.17.1.

Writers: anndata 0.12.19 with pandas 2.3.3 (W2) and with pandas 3.1.1 (W3).
`+ns` means `anndata.settings.allow_write_nullable_strings = True`.

| Fixture | Writer | Encodings beyond the baseline | 0.10.9 | 0.11.4 | 0.12.19 |
|---|---|---|---|---|---|
| dense, csr, csc | W2, W2+ns | none | OK | OK | OK |
| obs categorical | W2, W2+ns | none | OK | OK | OK |
| obs text column (repeated / unique values) | W2, W2+ns | none | OK | OK | OK |
| obs nullable Int64 / boolean | W2, W2+ns | none | OK | OK | OK |
| uns scalars, string, nested dict | W2, W2+ns | none | OK | OK | OK |
| uns value `None` | W2, W2+ns | `null` | FAIL | FAIL | OK |
| obs `string`-dtype column | W2 | write refused (RuntimeError) | - | - | - |
| obs `string`-dtype column | W2+ns | `nullable-string-array` (column) | FAIL | OK | OK |
| all eleven fixtures | W3 | write refused (RuntimeError) | - | - | - |
| ten fixtures without `None` | W3+ns | `nullable-string-array` (`obs/_index`, `var/_index`) | FAIL | OK | OK |
| uns value `None` | W3+ns | `null`, `nullable-string-array` | FAIL | FAIL | OK |

Totals: 21 files from W2 and 11 from W3, each opened by three readers (96
reads). Every outcome agrees with the table's prediction; every failure is
`IORegistryError: No read method registered for IOSpec(...)`.

Files produced by `write_portable` (dense; CSR with labels and coordinates;
empty CSR without compression) were opened by all three readers with identical
values, gene order and names.

Baseline encodings (opened by all three readers): `anndata 0.1.0`,
`dataframe 0.2.0`, `dict 0.1.0`, `array 0.2.0`, `csr_matrix 0.1.0`,
`csc_matrix 0.1.0`, `string-array 0.2.0`, `categorical 0.2.0`, `string 0.2.0`,
`numeric-scalar 0.2.0`, `nullable-integer 0.1.0`, `nullable-boolean 0.1.0`.

Not measured: anndata 0.13.x (needs Python 3.12+), anndata below 0.10,
awkward arrays, zarr stores.
