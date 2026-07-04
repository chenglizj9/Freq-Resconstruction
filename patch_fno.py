import re

def insert_pde_res(filename):
    with open(filename, "r") as f:
        code = f.read()

    # Import
    if "from physics_metric import evaluate_physics_residual" not in code:
        code = code.replace("import h5py\n", "import h5py\nfrom physics_metric import evaluate_physics_residual\n")
    
    # In evaluate_model loop:
    #                 obs_rmse = ...
    #                 unobs_rmse = ...
    # +               pde_res = evaluate_physics_residual(pred_i, ds.f if hasattr(ds, 'f') else None, int(sample_idx[i]), float(omega_raw[i]), meta["meta"])
    # wait, lrtfr uses `ds = _load_sparse_dataset(...)` which doesn't keep h5py.File handle.
    pass
