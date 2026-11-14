"""h5ad_compat: will a different anndata version be able to open this .h5ad?

A prediction file is opened on someone else's machine with ``anndata.read_h5ad``,
which reads *every* element in the file. One element whose on-disk encoding is
newer than the reader makes the whole read fail, even if that element is never
used. This module answers the question without importing anndata:

* ``scan(path)``            -- list every encoded element and HDF5 filter (h5py only)
* ``check(path, reader)``   -- verdict against a tested reader line ("0.10", "0.11", "0.12")
* ``write_portable(...)``   -- write a prediction using only the encodings every
                               tested reader opens, independent of the local
                               anndata / pandas versions

Unofficial community helper. It does not score, validate against a board, or
decide whether a file is acceptable for submission.

Copyright (c) 2026 StartupComplianceKit. MIT License.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import h5py
import numpy as np

__version__ = "1.0.0"

READERS = ("0.10", "0.11", "0.12")

# (encoding-type, encoding-version) -> oldest tested reader line that opens it.
# Every row was observed by reading synthetic files with anndata 0.10.9, 0.11.4
# and 0.12.19 (see MATRIX.md). Anything not listed is reported as "untested".
MIN_READER = {
    ("anndata", "0.1.0"): "0.10",
    ("dataframe", "0.2.0"): "0.10",
    ("dict", "0.1.0"): "0.10",
    ("array", "0.2.0"): "0.10",
    ("csr_matrix", "0.1.0"): "0.10",
    ("csc_matrix", "0.1.0"): "0.10",
    ("string-array", "0.2.0"): "0.10",
    ("categorical", "0.2.0"): "0.10",
    ("string", "0.2.0"): "0.10",
    ("numeric-scalar", "0.2.0"): "0.10",
    ("nullable-integer", "0.1.0"): "0.10",
    ("nullable-boolean", "0.1.0"): "0.10",
    ("nullable-string-array", "0.1.0"): "0.11",
    ("null", "0.1.0"): "0.12",
}

# HDF5 filters compiled into the HDF5 library itself.
BUILTIN_FILTERS = {1: "gzip", 2: "shuffle", 3: "fletcher32", 4: "szip", 5: "nbit", 6: "scaleoffset"}
LZF_FILTER = 32000  # bundled with h5py; other readers may need a plugin

# Elements the public scorer source reads after loading the file.
SCORED = ("X", "obs/_index", "var/_index", "obs/celltype", "obsm/spatial_3D")


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return None if value is None else str(value)


def _rank(reader):
    return READERS.index(reader)


def _is_scored(name, index_paths):
    if name in index_paths:
        return True
    return any(name == s or name.startswith(s + "/") for s in SCORED)


def scan(path):
    """Return a JSON-serialisable description of the file's encodings and filters."""
    elements, filters, problems, untagged = [], [], [], []
    with h5py.File(path, "r") as f:
        index_paths = set()
        for axis in ("obs", "var"):
            if axis in f and isinstance(f[axis], h5py.Group):
                index_paths.add(axis + "/" + (_text(f[axis].attrs.get("_index")) or "_index"))

        def add(name, obj):
            etype = _text(obj.attrs.get("encoding-type"))
            if etype is not None:
                elements.append({
                    "path": name or "/",
                    "encoding_type": etype,
                    "encoding_version": _text(obj.attrs.get("encoding-version")),
                    "scored": _is_scored(name, index_paths),
                })
            elif name:
                parent = obj.parent
                sparse_member = (_text(parent.attrs.get("encoding-type")) in ("csr_matrix", "csc_matrix")
                                 and name.rsplit("/", 1)[-1] in ("data", "indices", "indptr"))
                if not sparse_member:
                    untagged.append(name)
            if isinstance(obj, h5py.Dataset):
                plist = obj.id.get_create_plist()
                for i in range(plist.get_nfilters()):
                    code, _flags, _values, fname = plist.get_filter(i)
                    if code not in BUILTIN_FILTERS:
                        filters.append({"path": name, "filter_id": int(code),
                                        "filter_name": _text(fname) or ""})

        add("", f)
        f.visititems(add)

        legacy = "encoding-type" not in f.attrs
        for key in ("X", "obs", "var"):
            if key not in f:
                problems.append(f"missing '{key}': the file is incomplete "
                                "(an interrupted or failed write leaves files like this)")
        for p in sorted(index_paths):
            if p.split("/")[0] in f and p not in f:
                problems.append(f"missing '{p}': no row/column names stored")
    return {"elements": elements, "filters": filters, "problems": problems, "legacy_layout": legacy, "untagged": untagged}


def check(path, reader="0.10"):
    """Compare a file with a tested reader line. Returns a dict with 'verdict'.

    verdict is "pass", "fail" (something the reader is known not to open, or an
    incomplete file) or "unknown" (only untested encodings/filters were found).
    """
    if reader not in READERS:
        raise ValueError(f"reader must be one of {READERS}, got {reader!r}")
    report = scan(path)
    blocking, untested, notes = [], [], []
    for e in report["elements"]:
        need = MIN_READER.get((e["encoding_type"], e["encoding_version"]))
        label = f"{e['path']} [{e['encoding_type']} {e['encoding_version']}]"
        where = "scored element" if e["scored"] else "extra element"
        if need is None:
            untested.append(f"{label}: encoding not in the tested table ({where})")
        elif _rank(need) > _rank(reader):
            fix = ("rewrite it with a baseline encoding" if e["scored"]
                   else "omit it only if optional, or rewrite it with a baseline encoding")
            blocking.append(f"{label}: needs anndata >= {need} ({where}; {fix})")
    for flt in report["filters"]:
        label = f"{flt['path']} [HDF5 filter {flt['filter_id']} {flt['filter_name']}]".replace(" ]", "]")
        if flt["filter_id"] == LZF_FILTER:
            notes.append(f"{label}: LZF is bundled with h5py; other HDF5 readers may need the matching filter")
        else:
            untested.append(f"{label}: not a built-in HDF5 filter; a reading machine without "
                            "the matching plugin may be unable to decode it (use gzip or no compression)")
    if report["legacy_layout"]:
        untested.append("no root encoding-type attribute: pre-0.8 layout or not written by anndata; "
                     "element encodings could not be compared")
    elif report["untagged"]:
        untested.extend(f"{p}: no encoding-type attribute; legacy fallback not tested"
                        for p in report["untagged"])
    blocking = report["problems"] + blocking
    verdict = "fail" if blocking else ("unknown" if untested else "pass")
    return {"file": os.path.basename(str(path)), "reader": reader, "verdict": verdict,
            "blocking": blocking, "untested": untested, "notes": notes,
            "elements": report["elements"], "filters": report["filters"]}


def format_report(result):
    lines = [f"{result['file']}: {result['verdict'].upper()} for anndata {result['reader']}.x readers "
             f"({len(result['elements'])} encoded elements)"]
    for key, tag in (("blocking", "BLOCKING"), ("untested", "UNTESTED"), ("notes", "note")):
        lines += [f"  {tag}: {item}" for item in result[key]]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Portable writer
# ----------------------------------------------------------------------------

def _names(values, what, n=None):
    out = [v.decode("utf-8") if isinstance(v, bytes) else v for v in list(values)]
    if not all(isinstance(v, str) and v for v in out):
        raise ValueError(f"{what} must be non-empty strings")
    if n is not None and len(out) != n:
        raise ValueError(f"{what} has {len(out)} entries, expected {n}")
    return out


def _enc(obj, etype, version):
    obj.attrs["encoding-type"] = etype
    obj.attrs["encoding-version"] = version


def _write_strings(group, name, values):
    d = group.create_dataset(name, data=np.array(values, dtype=object), dtype=h5py.string_dtype("utf-8"))
    _enc(d, "string-array", "0.2.0")


def _write_frame(f, name, index, columns=()):
    g = f.create_group(name)
    _enc(g, "dataframe", "0.2.0")
    g.attrs["_index"] = "_index"
    g.attrs.create("column-order", np.array(list(columns), dtype=object), dtype=h5py.string_dtype("utf-8"))
    _write_strings(g, "_index", index)
    return g


def write_portable(path, X, var_names, obs_names=None, celltype=None, spatial_3d=None,
                   compression="gzip"):
    """Write a prediction .h5ad using only baseline encodings.

    X          Real 2-D array-like, or canonical scipy CSR (stored sparse). Saved as float32.
    var_names  gene names in the order you intend to submit; written verbatim.
    obs_names  optional cell names; defaults to "0", "1", ...
    celltype   optional per-cell labels, stored as a categorical in obs["celltype"].
    spatial_3d optional (n_cells, >=3) coordinates, stored in obsm["spatial_3D"].
    compression "gzip" or None. No other filter is offered on purpose.

    The file is written next to ``path`` under a temporary name and moved into
    place only after it has been closed, so a failed write never leaves a
    half-written file at ``path``.
    """
    if compression not in ("gzip", None):
        raise ValueError("compression must be 'gzip' or None")
    is_csr = getattr(X, "format", None) == "csr"
    if is_csr:
        X = X.copy()
        X.check_format(full_check=True)
        if len(X.shape) != 2 or not X.has_canonical_format:
            raise ValueError("sparse X must be 2-D canonical CSR (sorted, without duplicate coordinates)")
        if X.dtype.kind not in "biuf":
            raise ValueError("X must contain real numeric values")
        shape = tuple(int(s) for s in X.shape)
        data = np.asarray(X.data, dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError("X contains NaN or infinite values")
    elif hasattr(X, "format"):
        raise ValueError("sparse X must be CSR; convert with .tocsr() first")
    else:
        dense = np.asarray(X)
        if dense.ndim != 2:
            raise ValueError(f"X must be 2-D (cells x genes), got shape {dense.shape}")
        if dense.dtype.kind not in "biuf":
            raise ValueError("X must contain real numeric values")
        if not np.isfinite(dense).all():
            raise ValueError("X contains NaN or infinite values")
        dense = dense.astype(np.float32)
        if not np.isfinite(dense).all():
            raise ValueError("X overflows float32")
        shape = dense.shape
    n_obs, n_var = shape
    var_names = _names(var_names, "var_names", n_var)
    obs_names = [str(i) for i in range(n_obs)] if obs_names is None else _names(obs_names, "obs_names", n_obs)
    if celltype is not None:
        celltype = _names(celltype, "celltype", n_obs)
    if spatial_3d is not None:
        spatial_3d = np.asarray(spatial_3d)
        if spatial_3d.dtype.kind not in "biuf":
            raise ValueError("spatial_3d must contain real numeric values")
        spatial_3d = spatial_3d.astype(np.float32)
        if spatial_3d.ndim != 2 or spatial_3d.shape[0] != n_obs or spatial_3d.shape[1] < 3:
            raise ValueError(f"spatial_3d must have shape ({n_obs}, >=3), got {spatial_3d.shape}")
        if not np.isfinite(spatial_3d).all():
            raise ValueError("spatial_3d contains NaN or infinite values")

    kw = {"compression": "gzip", "compression_opts": 4} if compression else {}
    path = os.fspath(path)
    fd, tmp = tempfile.mkstemp(suffix=".h5ad.tmp", dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as f:
            _enc(f, "anndata", "0.1.0")
            if is_csr:
                g = f.create_group("X")
                _enc(g, "csr_matrix", "0.1.0")
                g.attrs["shape"] = np.array(shape, dtype=np.int64)
                g.create_dataset("data", data=data, **(kw if data.size else {}))
                g.create_dataset("indices", data=np.asarray(X.indices, dtype=np.int64),
                                 **(kw if data.size else {}))
                g.create_dataset("indptr", data=np.asarray(X.indptr, dtype=np.int64))
            else:
                d = f.create_dataset("X", data=dense, **(kw if dense.size else {}))
                _enc(d, "array", "0.2.0")
            obs = _write_frame(f, "obs", obs_names, ["celltype"] if celltype is not None else [])
            if celltype is not None:
                cats = sorted(set(celltype))
                lookup = {c: i for i, c in enumerate(cats)}
                c = obs.create_group("celltype")
                _enc(c, "categorical", "0.2.0")
                c.attrs["ordered"] = False
                codes = c.create_dataset("codes", data=np.array([lookup[v] for v in celltype], dtype=np.int32))
                _enc(codes, "array", "0.2.0")
                _write_strings(c, "categories", cats)
            _write_frame(f, "var", var_names)
            for name in ("obsm", "varm", "layers", "obsp", "varp", "uns"):
                _enc(f.create_group(name), "dict", "0.1.0")
            if spatial_3d is not None:
                d = f["obsm"].create_dataset("spatial_3D", data=spatial_3d)
                _enc(d, "array", "0.2.0")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="h5ad_compat",
        description="Check .h5ad encoding support for tested anndata readers (HDF5 metadata only).")
    ap.add_argument("files", nargs="+", help=".h5ad file(s) to check")
    ap.add_argument("--reader", default="0.10", choices=READERS,
                    help="oldest anndata line the file must open in (default: 0.10)")
    ap.add_argument("--json", action="store_true", help="print a JSON report")
    args = ap.parse_args(argv)
    results = []
    for p in args.files:
        try:
            results.append(check(p, args.reader))
        except OSError as exc:
            results.append({"file": os.path.basename(p), "reader": args.reader, "verdict": "fail",
                            "blocking": [f"not a readable HDF5 file ({type(exc).__name__})"],
                            "untested": [], "notes": [], "elements": [], "filters": []})
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n".join(format_report(r) for r in results))
    verdicts = {r["verdict"] for r in results}
    return 1 if "fail" in verdicts else (2 if "unknown" in verdicts else 0)


if __name__ == "__main__":
    sys.exit(main())
