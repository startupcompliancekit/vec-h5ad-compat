"""Synthetic end-to-end demo: write the same invented prediction two ways and
check which anndata reader lines can open each file.

    python demo.py

All numbers are random and generated here; no challenge data are used.

Copyright (c) 2026 StartupComplianceKit. MIT License.
"""
import os
import tempfile
import warnings

import anndata as ad
import numpy as np
import pandas as pd

from h5ad_compat import READERS, check, format_report, write_portable


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(2026)
    n_cells, n_genes = 300, 40
    X = np.log1p(rng.poisson(0.6, size=(n_cells, n_genes))).astype(np.float32)
    genes = [f"Gene{i:03d}" for i in rng.permutation(n_genes)]  # deliberately not sorted
    cells = [f"cell_{i:04d}" for i in range(n_cells)]
    labels = rng.choice(["TypeA", "TypeB", "TypeC"], size=n_cells).tolist()
    coords = rng.normal(size=(n_cells, 3)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        # 1. An ordinary-looking anndata write that carries two convenient extras:
        #    a free-text obs column with the pandas string dtype and uns["note"] = None.
        ordinary = os.path.join(tmp, "ordinary.h5ad")
        obs = pd.DataFrame({"celltype": pd.Categorical(labels),
                            "barcode_note": pd.array(cells, dtype="string")}, index=cells)
        a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes),
                       obsm={"spatial_3D": coords}, uns={"note": None})
        ad.settings.allow_write_nullable_strings = True
        a.write_h5ad(ordinary)

        # 2. The same prediction through the portable writer.
        portable = os.path.join(tmp, "portable.h5ad")
        write_portable(portable, X, genes, obs_names=cells, celltype=labels, spatial_3d=coords)

        for path in (ordinary, portable):
            for reader in READERS:
                print(format_report(check(path, reader)))
            print()

        back = ad.read_h5ad(portable)
        same = (np.array_equal(np.asarray(back.X), X) and list(back.var_names) == genes
                and list(back.obs_names) == cells and list(back.obs["celltype"]) == labels
                and np.array_equal(back.obsm["spatial_3D"], coords))
        print(f"portable.h5ad read back with anndata {ad.__version__}: "
              f"{'identical X, gene order, cell names, labels and coordinates' if same else 'MISMATCH'}")
        return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
