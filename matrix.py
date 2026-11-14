"""Reproduce the reader matrix in MATRIX.md with whatever anndata is installed.

    python matrix.py write fixtures/     # write one tiny synthetic file per encoding
    python matrix.py read  fixtures/     # try anndata.read_h5ad on each, print a table

The write output directory must not already exist.
Run "write" in the newest environment you have and "read" in each environment
you want to test. All data are random numbers generated here.

Copyright (c) 2026 StartupComplianceKit. MIT License.
"""
import glob
import os
import sys
import warnings

import numpy as np


def _cases():
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    X = rng.random((6, 4)).astype(np.float32)
    cells = [f"c{i}" for i in range(6)]
    genes = [f"g{i}" for i in range(4)]

    def make(X=X, obs=None, uns=None):
        obs = pd.DataFrame(index=pd.Index(cells)) if obs is None else pd.DataFrame(obs, index=cells)
        return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(genes)), uns=uns or {})

    return {
        "dense": lambda: make(),
        "csr": lambda: make(X=sp.csr_matrix(X)),
        "csc": lambda: make(X=sp.csc_matrix(X)),
        "obs_categorical": lambda: make(obs={"celltype": pd.Categorical(list("aabbcc"))}),
        "obs_text_repeated": lambda: make(obs={"celltype": list("aabbcc")}),
        "obs_text_unique": lambda: make(obs={"note": list("abcdef")}),
        "obs_string_dtype_unique": lambda: make(obs={"note": pd.array(list("abcdef"), dtype="string")}),
        "obs_nullable_int": lambda: make(obs={"k": pd.array([1, None, 3, 4, 5, 6], dtype="Int64")}),
        "obs_nullable_bool": lambda: make(obs={"k": pd.array([True, None, False, True, True, False], dtype="boolean")}),
        "uns_scalars": lambda: make(uns={"a": {"b": 1, "c": "x", "d": np.arange(3)}}),
        "uns_none": lambda: make(uns={"note": None}),
    }


def write(out):
    import anndata as ad

    os.makedirs(out, exist_ok=False)
    has_flag = hasattr(getattr(ad, "settings", None), "allow_write_nullable_strings")
    for name, build in _cases().items():
        for flag in ((False, True) if has_flag else (False,)):
            if has_flag:
                ad.settings.allow_write_nullable_strings = flag
            path = os.path.join(out, f"{name}{'__nullable_strings' if flag else ''}.h5ad")
            try:
                build().write_h5ad(path)
                print(f"written     {os.path.basename(path)}")
            except Exception as exc:  # the failure itself is a result worth seeing
                if os.path.exists(path):
                    os.remove(path)
                print(f"WRITE FAIL  {os.path.basename(path)}  {type(exc).__name__}")


def read(out):
    import anndata as ad
    from h5ad_compat import MIN_READER, scan

    print(f"reader: anndata {ad.__version__}")
    for path in sorted(glob.glob(os.path.join(out, "*.h5ad"))):
        try:
            ad.read_h5ad(path)
            status = "OK  "
        except Exception as exc:
            status = "FAIL"
            detail = type(exc).__name__
        elements = scan(path)["elements"]
        newest = sorted({MIN_READER.get((e["encoding_type"], e["encoding_version"]), "untested")
                         for e in elements} or {"untested"})[-1]
        special = sorted({e["encoding_type"] for e in elements
                          if MIN_READER.get((e["encoding_type"], e["encoding_version"])) != "0.10"})
        print(f"{status} {os.path.basename(path):48s} table says >= {newest:8s} {','.join(special)}"
              + (f"  ({detail})" if status == "FAIL" else ""))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    if len(sys.argv) != 3 or sys.argv[1] not in ("write", "read"):
        sys.exit(__doc__)
    {"write": write, "read": read}[sys.argv[1]](sys.argv[2])
