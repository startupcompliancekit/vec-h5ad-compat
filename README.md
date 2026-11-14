# h5ad-compat: will another anndata version open my prediction file?

A small, unofficial helper for Virtual Embryo Challenge participants.

## The problem

A prediction `.h5ad` is opened somewhere other than where it was written. The
public scorer source (`score_h5ad.py` in `aristoteleo/veckit`, commit
`cc3859a`) loads a prediction with `anndata.read_h5ad(path)` and its
`pyproject.toml` lists `anndata` without a version. `read_h5ad` reads **every**
element of the file, not only the ones that are scored. If one element uses an
on-disk encoding newer than the reading anndata, the whole read stops with

```
IORegistryError: No read method registered for IOSpec(encoding_type='null', encoding_version='0.1.0') ...
```

even when that element is a leftover `uns` entry nobody uses. Community READMEs
for this challenge report anndata versions from 0.11 to 0.13 and pandas 2 and 3,
so writer and reader versions do differ in practice. Three cases measured here
(details in [MATRIX.md](MATRIX.md)):

| What you did | What lands in the file | Oldest tested anndata that reads it |
|---|---|---|
| `adata.uns["anything"] = None`, written with anndata 0.12 | `null` element | 0.12 (0.10.9 and 0.11.4 fail) |
| a pandas `string`-dtype column, with `allow_write_nullable_strings = True` | `nullable-string-array` | 0.11 (0.10.9 fails) |
| all eleven tested fixtures with pandas 3.1.1 + anndata 0.12.19: fail by default; after switching `allow_write_nullable_strings` on, `obs_names` and `var_names` themselves | `nullable-string-array` in `obs/_index` and `var/_index` | 0.11 (0.10.9 fails) |

A successful local validation or round trip with the writing version of
anndata does not establish compatibility with an older reader.

## What is here

| File | Purpose |
|---|---|
| `h5ad_compat.py` | One module, needs only `h5py` and `numpy`. `check()` / CLI reads HDF5 metadata and compares its encoding tags with support measured in anndata 0.10.9, 0.11.4 and 0.12.19, identifying incompatible elements. `write_portable()` writes a prediction using only encodings all three open, whatever anndata/pandas you have installed. |
| `demo.py` | Synthetic end-to-end example. |
| `matrix.py` | Reproduce the reader matrix with your own anndata versions. |
| `MATRIX.md` | The measured matrix the lookup table is built from. |
| `test_h5ad_compat.py` | 25 tests. |

Who needs it: anyone whose file opens locally but who cannot control the
reader's anndata version; people on pandas 3; people writing files from a
pipeline that leaves `None` or free-text columns behind; and anyone who wants a
writer they can copy into a project (the module is a single MIT-licensed file).

## Five-minute quickstart

Python 3.11 or newer.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python demo.py
python -m unittest -v
```

`demo.py` writes one invented prediction twice and prints:

```
ordinary.h5ad: FAIL for anndata 0.10.x readers (20 encoded elements)
  BLOCKING: obs/barcode_note [nullable-string-array 0.1.0]: needs anndata >= 0.11 (extra element; omit it only if optional, or rewrite it with a baseline encoding)
  BLOCKING: uns/note [null 0.1.0]: needs anndata >= 0.12 (extra element; omit it only if optional, or rewrite it with a baseline encoding)
ordinary.h5ad: FAIL for anndata 0.11.x readers (20 encoded elements)
  BLOCKING: uns/note [null 0.1.0]: needs anndata >= 0.12 (extra element; omit it only if optional, or rewrite it with a baseline encoding)
ordinary.h5ad: PASS for anndata 0.12.x readers (20 encoded elements)

portable.h5ad: PASS for anndata 0.10.x readers (16 encoded elements)
portable.h5ad: PASS for anndata 0.11.x readers (16 encoded elements)
portable.h5ad: PASS for anndata 0.12.x readers (16 encoded elements)

portable.h5ad read back with anndata 0.12.19: identical X, gene order, cell names, labels and coordinates
```

### Check your own file

```bash
python h5ad_compat.py my_prediction.h5ad                 # against the oldest tested line (0.10)
python h5ad_compat.py my_prediction.h5ad --reader 0.11
python h5ad_compat.py a.h5ad b.h5ad --json
```

Exit status: `0` every file passes, `1` at least one file has a blocking
element or is incomplete, `2` untested encodings, filters or layouts were found without a known blocker.
The file is opened read-only; only HDF5 attributes and filter lists are read,
never expression values. Reports contain the file name without its directory.

Besides encodings the check reports:

- **incomplete files** (no `X`, `obs`, `var` or index). A failed
  `write_h5ad` can leave such a file behind, and it may still open as an empty
  AnnData;
- **HDF5 filters that are not built into HDF5** (for example zstd or blosc from
  a plugin package), which a reader without the plugin may be unable to decode;
  LZF is reported as a note because it is bundled with h5py.

Each finding says whether it sits on an element the public scorer source reads
(`X`, `obs_names`, `var_names`, `obs["celltype"]`, `obsm["spatial_3D"]`) or on
another element. Omit an element only if it is optional for your application;
otherwise rewrite it using supported encodings.

### Write a portable file

```python
from h5ad_compat import write_portable

write_portable(
    "prediction.h5ad",
    X,                       # real 2-D array or canonical scipy CSR; stored as float32
    var_names,               # gene names, written in exactly this order
    obs_names=None,          # optional; defaults to "0", "1", ...
    celltype=None,           # optional labels -> obs["celltype"] (categorical)
    spatial_3d=None,         # optional (n_cells, >=3) -> obsm["spatial_3D"]
    compression="gzip",      # or None
)
```

It uses h5py directly, so the output does not depend on your anndata or pandas
version. It refuses NaN/inf, float32 overflow, wrong lengths, empty or
non-string names, complex/non-numeric values, malformed CSR, and sparse input
that is not canonical CSR (sorted indices, no duplicate coordinates), and it writes to a temporary file
first, so a failed call never replaces or truncates an existing file. It does
**not** reorder genes, check a board's gene panel or cell-count range, or
change values other than the cast to float32. Run your usual format validator
afterwards.

## Scope and limits

- This is not an official tool, not a validator and not a scorer. A `PASS`
  means "no inspected encoding tag or filter is outside the tested table for
  that tested reader version". This is a metadata check, not a full read test:
  malformed payloads, schema/type mismatches and broken links are not validated.
  PASS does not guarantee the file can be decoded, is valid for a board or will
  be accepted. Test with `anndata.read_h5ad` in the target environment as well.
- The anndata version on the scoring host is not published as far as we know.
  The check identifies known encoding barriers; `--reader 0.10` selects
  the oldest tested reader. It does not establish the scoring host's behavior.
- "Tested" means anndata 0.10.9, 0.11.4 and 0.12.19, reading files written by
  anndata 0.12.19 with pandas 2.3.3 and pandas 3.1.1 (Python 3.11, h5py 3.16.0).
  anndata 0.13 and anndata older than 0.10 were not run; encodings outside the
  table are reported as `UNTESTED`, never as passing or failing.
- Files in the pre-0.8 layout (no root encoding tag), and untagged modern
  elements outside CSR/CSC storage arrays, are `UNKNOWN`, not assumed portable.
- Only `.h5ad` (HDF5) is covered, not zarr.
- `write_portable` stores `X`, names, optional labels and optional coordinates
  only. Layers, `raw`, `uns`, `obsp` and other `obs`/`var` columns are out of
  scope.
- All example and test data are random numbers generated by the scripts. No
  challenge data are included or needed.

## License

MIT. Copyright (c) 2026 StartupComplianceKit.

## Sources

- [Challenge community award](https://virtualembryo.ai/challenge/community)
- [Public scorer snapshot](https://github.com/aristoteleo/veckit/tree/cc3859a)
- [AnnData on-disk format](https://anndata.readthedocs.io/en/stable/fileformat-prose.html)

To repeat the measured matrix, create a **new** fixture directory with
`python matrix.py write fixtures`, then run `python matrix.py read fixtures`
using each reader environment. `write` refuses an existing directory. Keep
separate directories for each writer/pandas combination.
