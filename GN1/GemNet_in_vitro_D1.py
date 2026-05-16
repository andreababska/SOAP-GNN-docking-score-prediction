#!/usr/bin/env python3
import os
import gc
import time

# CUDA pamatovy alokator: redukuje fragmentaciu pri velkych autograd grafoch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from ase.io import read
from gemnet.model.gemnet import GemNet
from gemnet.training.data_container import DataContainer


# Compatibility for the original GemNet DataContainer on newer NumPy.
if not hasattr(np, "bool"):
    np.bool = np.bool_


# =========================================================
# PARAMETER
# =========================================================

FOLD = 1

model_name  = f"GN{FOLD}_official.pt"
data_file   = "../../../in_vitro.xyz"
output_file = f"in_vitro_GN{FOLD}.csv"
time_file   = "in_vitro_test_times.csv"

# DOLEZITE: hyperparametre musia presne sediet s in_vivo trenovacim skriptom,
# inak model.load_state_dict() zlyha.
cutoff = 7.0
int_cutoff = cutoff
triplets_only = True
batch_size = 16


# =========================================================
# DataContainer wrapper (same as in training script)
# =========================================================
class ASEAtomsDataContainer(DataContainer):
    """DataContainer-compatible wrapper for ASE Atoms lists.

    Inherits batching/index construction from the official GemNet DataContainer.
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
# 1. DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Pouzivam device:", device)
print(f"PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")
print(f"batch_size = {batch_size}")
print(f"cutoff = {cutoff}")

if not os.path.exists("scale.json"):
    with open("scale.json", "w") as f:
        f.write("{}")


# =========================================================
# 2. LOAD MODEL
# =========================================================
# Architektura MUSI presne sediet s in_vivo trenovacim skriptom:
# num_spherical=4, num_radial=6, num_atom=3, cutoff=7.0, direct_forces=False.
print(f"Nacitavam model {model_name}")
model = GemNet(
    num_spherical=4,
    num_radial=6,
    num_blocks=3,
    emb_size_atom=128,
    emb_size_edge=64,
    emb_size_trip=64,
    emb_size_quad=32,
    emb_size_rbf=16,
    emb_size_cbf=16,
    emb_size_sbf=32,
    emb_size_bil_quad=32,
    emb_size_bil_trip=64,
    num_before_skip=1,
    num_after_skip=2,
    num_concat=1,
    num_atom=3,
    triplets_only=triplets_only,
    num_targets=1,
    direct_forces=False,
    cutoff=cutoff,
    int_cutoff=int_cutoff,
    scale_file="scale.json",
    extensive=False,
).to(device)
model.load_state_dict(torch.load(model_name, map_location=device))
model.eval()


# =========================================================
# 3. LOAD DATA + BUILD DATACONTAINER
# =========================================================
print(f"Nacitavam {data_file}...")
t0 = time.time()
all_structures = read(data_file, index=":")
n_total = len(all_structures)
print(f"Pocet molekul v in_vitro: {n_total}")

print("Vytvaram official GemNet DataContainer wrapper")
test_data = ASEAtomsDataContainer(
    all_structures,
    cutoff=cutoff,
    int_cutoff=int_cutoff,
    triplets_only=triplets_only,
)
data_time = time.time() - t0
print(f"DataContainer pripraveny za {data_time:.1f} s")


# =========================================================
# 4. PREDICTION
# =========================================================
csv_f = open(output_file, "w")
csv_f.write("global_index;name;Predicted\n")

print("Robim predikciu...")
total_start = time.time()
pred_time_total = 0.0

n_batches = (n_total + batch_size - 1) // batch_size
log_every = max(1, n_batches // 20)

# GemNet with direct_forces=False needs gradients in eval too
# (forces are computed as -dE/dR via autograd).
with torch.set_grad_enabled(True):
    for batch_idx, start_idx in enumerate(range(0, n_total, batch_size)):
        end_idx = min(start_idx + batch_size, n_total)
        batch_ids = list(range(start_idx, end_idx))

        t_pred = time.time()
        batch = test_data[batch_ids]
        inputs, _targets = split_inputs_targets(batch)
        inputs = to_device(inputs, device)

        E_mol, _F_pred = model(inputs)
        preds = E_mol.detach().cpu().numpy().ravel()
        pred_time_total += time.time() - t_pred

        # write to CSV immediately
        for global_idx, y in zip(batch_ids, preds):
            name = all_structures[global_idx].info.get("name", "unknown")
            csv_f.write(f"{global_idx};{name};{y}\n")

        # cleanup
        del batch, inputs, E_mol, _F_pred, preds
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (batch_idx + 1) % log_every == 0 or batch_idx + 1 == n_batches:
            print(f"  batch {batch_idx + 1}/{n_batches}  ({end_idx}/{n_total})")

csv_f.flush()
os.fsync(csv_f.fileno())
csv_f.close()

total_time = time.time() - total_start


# =========================================================
# 5. TIMING LOG
# =========================================================
print("\n=== HOTOVO ===")
print(f"Model:        {model_name}")
print(f"Data:         {data_file}")
print(f"Pocet molekul: {n_total}")
print(f"Celkovy cas:  {total_time:.1f} s ({total_time/60:.2f} min)")
print(f"Data cas:     {data_time:.1f} s")
print(f"Prediction:   {pred_time_total:.1f} s")
print(f"Vystup:       {output_file}")

if not os.path.exists(time_file):
    with open(time_file, "w") as f:
        f.write(
            "Model;Fold;"
            "Total_seconds;Total_minutes;"
            "Data_seconds;Prediction_seconds;"
            "Num_molecules\n"
        )

with open(time_file, "a") as f:
    f.write(
        f"GemNet;Fold {FOLD};"
        f"{total_time:.2f};{total_time/60:.2f};"
        f"{data_time:.2f};{pred_time_total:.2f};"
        f"{n_total}\n"
    )

print(f"Cas ulozeny do {time_file}")