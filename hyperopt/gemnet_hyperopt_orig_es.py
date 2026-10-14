# =========================================================
# LIBRARIES
# =========================================================
import gc
import os
import random
import traceback

import numpy as np
import pandas as pd
import torch
from ase.io import read
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split

from gemnet.model.gemnet import GemNet
from gemnet.training.data_container import DataContainer


# Compatibility for the original GemNet DataContainer on newer NumPy.
if not hasattr(np, "bool"):
    np.bool = np.bool_


BEST_GLOBAL_MSE = float("inf")

if not os.path.exists("scale.json"):
    with open("scale.json", "w") as f:
        f.write("{}")

batch_size = 32
epochs = 50
max_evals = 50
patience = 10
min_delta = 1e-3

# =========================================================
# LOAD DATA
# =========================================================
file_name = "../../in_vivo.xyz"
structures = read(file_name, index=":10000")
y = np.array([float(s.get_potential_energy()) for s in structures], dtype=np.float32)

print("Loaded molecules:", len(structures))

max_atomic_number = max(int(max(s.get_atomic_numbers())) for s in structures)
print(f"Max atomic number: {max_atomic_number}")


# =========================================================
# DataContainer wrapper for ASE Atoms
# =========================================================
class ASEAtomsDataContainer(DataContainer):
    """DataContainer-compatible wrapper for ASE Atoms lists.

    Inherits batching and edge/triplet indexing from the official GemNet
    DataContainer instead of building them by hand.
    """

    def __init__(self, structures, cutoff, int_cutoff, triplets_only=True):
        self.index_keys = [
            "batch_seg",
            "id_undir",
            "id_swap",
            "id_c",
            "id_a",
            "id3_expand_ba",
            "id3_reduce_ca",
            "Kidx3",
        ]
        if not triplets_only:
            self.index_keys += [
                "id4_int_b",
                "id4_int_a",
                "id4_reduce_ca",
                "id4_expand_db",
                "id4_reduce_cab",
                "id4_expand_abd",
                "Kidx4",
                "id4_reduce_intm_ca",
                "id4_expand_intm_db",
                "id4_reduce_intm_ab",
                "id4_expand_intm_ab",
            ]

        self.triplets_only = triplets_only
        self.cutoff = cutoff
        self.int_cutoff = int_cutoff
        self.addID = False
        self.keys = ["N", "Z", "R", "F", "E"]
        self.transforms = []
        self.targets = ["E", "F"]

        self.N = np.array([len(atoms) for atoms in structures], dtype=np.int32)
        self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        n_atoms_total = int(self.N_cumsum[-1])

        self.Z = np.empty(n_atoms_total, dtype=np.int32)
        self.R = np.empty((n_atoms_total, 3), dtype=np.float32)
        self.F = np.zeros((n_atoms_total, 3), dtype=np.float32)
        self.E = np.empty((len(structures), 1), dtype=np.float32)

        atom_start = 0
        for mol_idx, atoms in enumerate(structures):
            atom_end = atom_start + len(atoms)
            self.Z[atom_start:atom_end] = atoms.get_atomic_numbers()
            self.R[atom_start:atom_end] = atoms.get_positions().astype(np.float32)
            self.E[mol_idx, 0] = float(atoms.get_potential_energy())
            atom_start = atom_end

        self.dtypes, dtypes_target = self.get_dtypes()
        self.dtypes.update(dtypes_target)


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def split_inputs_targets(batch):
    inputs = {key: value for key, value in batch.items() if key not in ("E", "F")}
    targets = {key: value for key, value in batch.items() if key in ("E", "F")}
    return inputs, targets


# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================================================
# Train / val split (indices into structures)
# =========================================================
n = len(structures)
all_idx = np.arange(n)
train_idx, val_idx = train_test_split(all_idx, test_size=0.20, random_state=42)
print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")


# =========================================================
# Epoch loop using DataContainer indexing
# =========================================================
def run_epoch(idxs, dataset, model, optimizer, loss_fn, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    idxs = list(idxs)
    if train:
        random.shuffle(idxs)

    total_loss = 0.0
    n_samples = 0

    # GemNet with direct_forces=False needs gradients in eval too
    # (forces are computed as -dE/dR via autograd).
    with torch.set_grad_enabled(True):
        for start in range(0, len(idxs), batch_size):
            batch_ids = idxs[start:start + batch_size]
            batch = dataset[batch_ids]
            inputs, targets = split_inputs_targets(batch)
            inputs = to_device(inputs, device)
            y_batch = targets["E"].to(device)

            E_mol, F_pred = model(inputs)
            loss = loss_fn(E_mol, y_batch)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(batch_ids)
            n_samples += len(batch_ids)

            del batch, inputs, targets, y_batch, E_mol, F_pred, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return total_loss / n_samples



# =========================================================
# OBJECTIVE FUNCTION
# =========================================================
def objective(params):
    global BEST_GLOBAL_MSE

    mse = 1e6
    model = None
    optimizer = None
    loss_fn = None

    try:
        print("\nParams:", params)

        cutoff = float(params["cutoff"])
        dataset = ASEAtomsDataContainer(
        structures,
        cutoff=cutoff,
        int_cutoff=cutoff,
        triplets_only=True,
    		)


        model = GemNet(
            num_blocks=int(params["num_blocks"]),
            num_radial=int(params["num_radial"]),
            num_spherical=int(params["num_spherical"]),
            emb_size_atom=int(params["emb_size_atom"]),
            emb_size_edge=int(params["emb_size_edge"]),
            emb_size_trip=int(params["emb_size_trip"]),
            num_atom=int(params["num_atom"]),
            # Fixed params
            emb_size_quad=32,
            emb_size_rbf=16,
            emb_size_cbf=16,
            emb_size_sbf=32,
            emb_size_bil_quad=32,
            emb_size_bil_trip=64,
            num_before_skip=1,
            num_after_skip=2,
            num_concat=1,
            triplets_only=True,
            num_targets=1,
            direct_forces=False,
            cutoff=cutoff,
            int_cutoff=cutoff,
            scale_file="scale.json",
            extensive=False,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()
        best_val = float("inf")
        epochs_no_improve = 0

        for ep in range(1, epochs + 1):
            tr = run_epoch(train_idx, dataset, model, optimizer, loss_fn, train=True)
            va = run_epoch(val_idx,   dataset, model, optimizer, loss_fn, train=False)
            print(f"Epoch {ep:02d}/{epochs} | Train MSE: {tr:.6f} | Val MSE: {va:.6f}")

            if va < best_val - min_delta:
                best_val = va
                epochs_no_improve = 0
                torch.save(model.state_dict(), "best_gemnet.pt")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {ep} "
                    f"(no improvement for {patience} epochs). "
                    f"Best Val MSE: {best_val:.6f}"
                )
                break

        model.load_state_dict(torch.load("best_gemnet.pt", map_location=device))
        mse = run_epoch(val_idx, dataset, model, optimizer, loss_fn, train=False)

        if mse < BEST_GLOBAL_MSE:
            BEST_GLOBAL_MSE = mse
            torch.save(model.state_dict(), "gn_best_model.pt")
            print(f"New best model saved (MSE={mse:.6f})")

    except Exception:
        traceback.print_exc()
        mse = 1e6
    finally:
        try:
            del model
        except Exception:
            pass
        try:
            del optimizer
        except Exception:
            pass
        try:
            del loss_fn
        except Exception:
            pass
        try:
            del dataset
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"→ MSE: {mse:.6f}")
    return {"loss": mse, "status": STATUS_OK}


# =========================================================
# HYPERPARAMETER SPACE
# =========================================================
space = {
    "num_blocks":    hp.quniform("num_blocks", 2, 4, 1),
    "num_radial":    hp.quniform("num_radial", 4, 12, 1),
    "num_spherical": hp.quniform("num_spherical", 4, 12, 1),
    "emb_size_atom": hp.choice("emb_size_atom", [64, 128]),
    "emb_size_edge": hp.choice("emb_size_edge", [64, 128]),
    "emb_size_trip": hp.choice("emb_size_trip", [32, 64]),
    "cutoff":        hp.choice("cutoff", [5.0, 6.0, 7.0, 8.0]),
    "num_atom":      hp.choice("num_atom", [1, 3, 5]),
}

# =========================================================
# RUN OPTIMIZATION
# =========================================================
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=trials,
    rstate=np.random.default_rng(42),
)
print("\n=== BEST PARAMETERS ===")
print(best)


# =========================================================
# EXPORT RESULTS TO CSV
# =========================================================
def export_trials(trials):
    records = []

    emb_atom_vals = [64, 128]
    emb_edge_vals = [64, 128]
    emb_trip_vals = [32, 64]
    cutoff_vals   = [5.0, 6.0, 7.0, 8.0]
    num_atom_vals = [1, 3, 5]

    for t in trials.trials:
        v = t["misc"]["vals"]
        records.append({
            "num_blocks":      int(v["num_blocks"][0]),
            "num_radial":      int(v["num_radial"][0]),
            "num_spherical":   int(v["num_spherical"][0]),
            "emb_size_atom":   emb_atom_vals[int(v["emb_size_atom"][0])],
            "emb_size_edge":   emb_edge_vals[int(v["emb_size_edge"][0])],
            "emb_size_trip":   emb_trip_vals[int(v["emb_size_trip"][0])],
            "cutoff":          cutoff_vals[int(v["cutoff"][0])],
            "num_atom":        num_atom_vals[int(v["num_atom"][0])],

            "emb_size_quad":    32,
            "emb_size_rbf":     16,
            "emb_size_cbf":     16,
            "emb_size_sbf":     32,
            "emb_size_bil_quad":32,
            "emb_size_bil_trip":64,
            "num_before_skip":  1,
            "num_after_skip":   2,
            "num_concat":       1,
            "triplets_only":    True,

            "batch_size":    batch_size,
            "learning_rate": 0.001,
            "epochs":        epochs,
            "mse":           t["result"]["loss"],
        })

    df = pd.DataFrame(records).sort_values("mse")
    df.to_csv("GemNet_hyperopt_official.csv", sep=";", index=False)
    print("\nSaved: GemNet_hyperopt_official.csv")
    print(df.head(10).to_string(index=False))


export_trials(trials)
