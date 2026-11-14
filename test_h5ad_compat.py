"""Tests for h5ad_compat. Run:  python -m unittest -v

Copyright (c) 2026 StartupComplianceKit. MIT License.
"""
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
import warnings

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

import h5ad_compat as hc
import matrix


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class Base(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        rng = np.random.default_rng(7)
        self.X = rng.random((9, 5)).astype(np.float32)
        self.X[self.X < 0.4] = 0
        self.genes = ["Zeta", "alpha", "Mid1", "β-cat", "a b"]
        self.cells = [f"c{i}" for i in range(9)]
        self.labels = list("abcabcabc")
        self.coords = rng.normal(size=(9, 3)).astype(np.float32)

    def tearDown(self):
        self._tmp.cleanup()

    def p(self, name):
        return os.path.join(self.tmp, name)

    def anndata_file(self, name, obs=None, uns=None, nullable=False):
        obs = pd.DataFrame(index=self.cells) if obs is None else pd.DataFrame(obs, index=self.cells)
        a = ad.AnnData(X=self.X, obs=obs, var=pd.DataFrame(index=self.genes), uns=uns or {})
        ad.settings.allow_write_nullable_strings = nullable
        a.write_h5ad(self.p(name))
        return self.p(name)


class PortableWriter(Base):
    def test_dense_round_trip(self):
        path = hc.write_portable(self.p("d.h5ad"), self.X.astype(np.float64), self.genes)
        a = ad.read_h5ad(path)
        self.assertEqual(a.X.dtype, np.float32)
        np.testing.assert_array_equal(np.asarray(a.X), self.X)
        self.assertEqual(list(a.var_names), self.genes)
        self.assertEqual(list(a.obs_names), [str(i) for i in range(9)])

    def test_csr_round_trip_with_labels_and_coordinates(self):
        path = hc.write_portable(self.p("s.h5ad"), sp.csr_matrix(self.X), self.genes,
                                 obs_names=self.cells, celltype=self.labels, spatial_3d=self.coords)
        a = ad.read_h5ad(path)
        self.assertTrue(sp.issparse(a.X))
        np.testing.assert_array_equal(a.X.toarray(), self.X)
        self.assertEqual(list(a.obs_names), self.cells)
        self.assertEqual(list(a.obs["celltype"]), self.labels)
        self.assertEqual(str(a.obs["celltype"].dtype), "category")
        np.testing.assert_array_equal(a.obsm["spatial_3D"], self.coords)

    def test_gene_order_and_unicode_names_kept_verbatim(self):
        path = hc.write_portable(self.p("g.h5ad"), self.X, self.genes)
        with h5py.File(path, "r") as f:
            stored = [v.decode("utf-8") for v in f["var/_index"][:]]
        self.assertEqual(stored, self.genes)

    def test_empty_sparse_and_uncompressed(self):
        path = hc.write_portable(self.p("e.h5ad"), sp.csr_matrix((9, 5)), self.genes, compression=None)
        self.assertEqual(ad.read_h5ad(path).X.nnz, 0)

    def test_portable_files_pass_every_reader(self):
        path = hc.write_portable(self.p("s.h5ad"), sp.csr_matrix(self.X), self.genes,
                                 celltype=self.labels, spatial_3d=self.coords)
        for reader in hc.READERS:
            result = hc.check(path, reader)
            self.assertEqual((result["verdict"], result["blocking"], result["untested"]), ("pass", [], []))

    def test_only_gzip_filter_is_used(self):
        path = hc.write_portable(self.p("d.h5ad"), self.X, self.genes)
        self.assertEqual(hc.scan(path)["filters"], [])
        with h5py.File(path, "r") as f:
            self.assertEqual(f["X"].compression, "gzip")

    def test_rejects_bad_input(self):
        bad = self.X.copy()
        bad[0, 0] = np.nan
        cases = [
            lambda: hc.write_portable(self.p("x.h5ad"), bad, self.genes),
            lambda: hc.write_portable(self.p("x.h5ad"), self.X, self.genes[:-1]),
            lambda: hc.write_portable(self.p("x.h5ad"), self.X, self.genes[:-1] + [""]),
            lambda: hc.write_portable(self.p("x.h5ad"), self.X, self.genes[:-1] + [5]),
            lambda: hc.write_portable(self.p("x.h5ad"), self.X[0], self.genes),
            lambda: hc.write_portable(self.p("x.h5ad"), sp.csc_matrix(self.X), self.genes),
            lambda: hc.write_portable(self.p("x.h5ad"), self.X, self.genes, compression="lzf"),
            lambda: hc.write_portable(self.p("x.h5ad"), self.X, self.genes, spatial_3d=self.coords[:, :2]),
            lambda: hc.write_portable(self.p("x.h5ad"), self.X, self.genes, celltype=self.labels[:-1]),
            lambda: hc.write_portable(self.p("x.h5ad"), self.X.astype(np.float64) * 1e300, self.genes),
        ]
        for i, case in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                case()
        self.assertEqual(os.listdir(self.tmp), [])

    def test_rejects_complex_values_without_truncating_them(self):
        for X, coords in [(self.X.astype(complex) + 1j, None),
                          (sp.csr_matrix(self.X.astype(complex) + 1j), None),
                          (self.X, self.coords.astype(complex) + 1j)]:
            with self.subTest(sparse=sp.issparse(X), coordinates=coords is not None):
                with self.assertRaisesRegex(ValueError, "real numeric"):
                    hc.write_portable(self.p("complex.h5ad"), X, self.genes, spatial_3d=coords)
        self.assertEqual(os.listdir(self.tmp), [])

    def test_duplicate_csr_rejected_without_changing_input_or_destination(self):
        X = sp.csr_matrix((np.array([1e8, 1, -1e8]), [0, 0, 0], [0, 3]), shape=(1, 1))
        before = X.data.copy()
        path = hc.write_portable(self.p("keep.h5ad"), self.X, self.genes)
        digest = sha(path)
        with self.assertRaisesRegex(ValueError, "canonical CSR"):
            hc.write_portable(path, X, ["g"])
        np.testing.assert_array_equal(X.data, before)
        self.assertEqual(sha(path), digest)

    def test_malformed_csr_rejected(self):
        X = sp.csr_matrix(self.X)
        X.indices[0] = X.shape[1]
        with self.assertRaises(ValueError):
            hc.write_portable(self.p("bad.h5ad"), X, self.genes)
        self.assertEqual(os.listdir(self.tmp), [])

    def test_failed_write_keeps_existing_file(self):
        path = hc.write_portable(self.p("keep.h5ad"), self.X, self.genes)
        before = sha(path)
        with self.assertRaises(ValueError):
            hc.write_portable(path, self.X, self.genes, celltype=["ok"] * 8 + [""])
        self.assertEqual(sha(path), before)
        self.assertEqual(os.listdir(self.tmp), ["keep.h5ad"])


class Checker(Base):
    def test_plain_anndata_file_passes(self):
        path = self.anndata_file("plain.h5ad", obs={"celltype": pd.Categorical(self.labels)},
                                 uns={"a": {"b": 1, "c": "x"}})
        self.assertEqual(hc.check(path, "0.10")["verdict"], "pass")

    def test_uns_none_needs_0_12_and_is_an_extra_element(self):
        path = self.anndata_file("none.h5ad", uns={"note": None})
        self.assertEqual(hc.check(path, "0.12")["verdict"], "pass")
        for reader in ("0.10", "0.11"):
            result = hc.check(path, reader)
            self.assertEqual(result["verdict"], "fail")
            self.assertEqual(len(result["blocking"]), 1)
            self.assertIn("uns/note [null 0.1.0]: needs anndata >= 0.12 (extra element", result["blocking"][0])

    def test_nullable_string_column_needs_0_11(self):
        path = self.anndata_file("ns.h5ad", obs={"note": pd.array(self.cells, dtype="string")}, nullable=True)
        self.assertEqual(hc.check(path, "0.10")["verdict"], "fail")
        self.assertIn("obs/note [nullable-string-array 0.1.0]", hc.check(path, "0.10")["blocking"][0])
        self.assertEqual(hc.check(path, "0.11")["verdict"], "pass")
        self.assertEqual(hc.check(path, "0.12")["verdict"], "pass")

    def test_nullable_string_gene_index_is_reported_as_scored(self):
        path = hc.write_portable(self.p("idx.h5ad"), self.X, self.genes)
        with h5py.File(path, "a") as f:
            del f["var/_index"]
            g = f["var"].create_group("_index")
            g.attrs["encoding-type"], g.attrs["encoding-version"] = "nullable-string-array", "0.1.0"
            v = g.create_dataset("values", data=np.array(self.genes, dtype=object), dtype=h5py.string_dtype())
            v.attrs["encoding-type"], v.attrs["encoding-version"] = "string-array", "0.2.0"
            m = g.create_dataset("mask", data=np.zeros(5, dtype=bool))
            m.attrs["encoding-type"], m.attrs["encoding-version"] = "array", "0.2.0"
        self.assertEqual(list(ad.read_h5ad(path).var_names), self.genes)
        result = hc.check(path, "0.10")
        self.assertEqual(result["verdict"], "fail")
        self.assertIn("var/_index [nullable-string-array 0.1.0]: needs anndata >= 0.11 (scored element",
                      result["blocking"][0])

    def test_incomplete_file_fails(self):
        with h5py.File(self.p("half.h5ad"), "w") as f:
            f.attrs["encoding-type"], f.attrs["encoding-version"] = "anndata", "0.1.0"
            f.create_group("obs")
        result = hc.check(self.p("half.h5ad"), "0.12")
        self.assertEqual(result["verdict"], "fail")
        self.assertEqual(len(result["blocking"]), 3)

    def test_unknown_encoding_and_unknown_filter_are_untested(self):
        path = hc.write_portable(self.p("u.h5ad"), self.X, self.genes)
        with h5py.File(path, "a") as f:
            d = f["uns"].create_dataset("future", data=np.arange(3))
            d.attrs["encoding-type"], d.attrs["encoding-version"] = "array", "9.9.9"
            packed = f["uns"].create_dataset("packed", data=np.arange(64.0), chunks=(64,), compression=32015,
                                    compression_opts=(3,), allow_unknown_filter=True)
            packed.attrs["encoding-type"], packed.attrs["encoding-version"] = "array", "0.2.0"
        result = hc.check(path, "0.12")
        self.assertEqual(result["verdict"], "unknown")
        self.assertEqual(len(result["untested"]), 2)
        self.assertTrue(any("filter 32015" in u for u in result["untested"]))

    def test_lzf_is_a_note_not_a_failure(self):
        path = hc.write_portable(self.p("l.h5ad"), self.X, self.genes)
        with h5py.File(path, "a") as f:
            d = f["uns"].create_dataset("z", data=np.arange(64.0), compression="lzf")
            d.attrs["encoding-type"], d.attrs["encoding-version"] = "array", "0.2.0"
        result = hc.check(path, "0.10")
        self.assertEqual(result["verdict"], "pass")
        self.assertTrue(any("LZF" in n for n in result["notes"]))

    def test_legacy_layout_is_unknown(self):
        with h5py.File(self.p("old.h5ad"), "w") as f:
            f.create_dataset("X", data=self.X)
            for axis in ("obs", "var"):
                f.create_group(axis).create_dataset("_index", data=np.array([b"a"]))
        result = hc.check(self.p("old.h5ad"))
        self.assertEqual(result["verdict"], "unknown")
        self.assertTrue(any("pre-0.8 layout" in n for n in result["untested"]))

    def test_untagged_modern_element_is_unknown(self):
        path = hc.write_portable(self.p("missing-tag.h5ad"), self.X, self.genes)
        with h5py.File(path, "a") as f:
            del f["X"].attrs["encoding-type"]
        result = hc.check(path)
        self.assertEqual(result["verdict"], "unknown")
        self.assertTrue(any("X: no encoding-type" in n for n in result["untested"]))

    def test_check_does_not_modify_the_file_and_rejects_unknown_reader(self):
        path = self.anndata_file("ro.h5ad", uns={"note": None})
        before = sha(path)
        hc.check(path, "0.10")
        self.assertEqual(sha(path), before)
        with self.assertRaises(ValueError):
            hc.check(path, "0.9")

    def test_table_covers_what_this_anndata_writes(self):
        path = self.anndata_file("all.h5ad", obs={
            "celltype": pd.Categorical(self.labels), "i": pd.array([1, None] + [2] * 7, dtype="Int64"),
            "b": pd.array([True, None] + [False] * 7, dtype="boolean")}, uns={"s": "x", "n": 1.5, "v": np.arange(3)})
        self.assertEqual(hc.check(path, "0.10")["untested"], [])


class MatrixOutput(Base):
    def test_write_refuses_existing_directory_without_modifying_it(self):
        marker = self.p("dense.h5ad")
        with open(marker, "wb") as f:
            f.write(b"existing data")
        before = sha(marker)
        with self.assertRaises(FileExistsError):
            matrix.write(self.tmp)
        self.assertEqual(sha(marker), before)
        self.assertEqual(os.listdir(self.tmp), ["dense.h5ad"])


class CommandLine(Base):
    def run_cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = hc.main(list(args))
        return code, out.getvalue()

    def test_exit_codes_and_json_without_directories(self):
        good = hc.write_portable(self.p("good.h5ad"), self.X, self.genes)
        bad = self.anndata_file("bad.h5ad", uns={"note": None})
        self.assertEqual(self.run_cli(good)[0], 0)
        self.assertEqual(self.run_cli(bad)[0], 1)
        self.assertEqual(self.run_cli(bad, "--reader", "0.12")[0], 0)
        code, text = self.run_cli(good, bad, "--json")
        self.assertEqual(code, 1)
        self.assertEqual([r["file"] for r in json.loads(text)], ["good.h5ad", "bad.h5ad"])
        self.assertNotIn(self.tmp, text)

    def test_non_hdf5_input_is_reported(self):
        with open(self.p("junk.h5ad"), "wb") as fh:
            fh.write(b"not hdf5")
        code, text = self.run_cli(self.p("junk.h5ad"))
        self.assertEqual(code, 1)
        self.assertIn("not a readable HDF5 file", text)
        self.assertNotIn(self.tmp, text)


if __name__ == "__main__":
    unittest.main()
