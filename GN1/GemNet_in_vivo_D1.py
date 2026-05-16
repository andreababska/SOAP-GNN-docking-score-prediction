#!/usr/bin/env python3
import os
import gc
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read
from gemnet.model.gemnet import GemNet
from gemnet.training.data_container import DataContainer


# Compatibility for the original GemNet DataContainer on newer NumPy.
if not hasattr(np, "bool"):
    np.bool = np.bool_


FOLD = 1

train_file = f"in_vivo_0{FOLD}_train.xyz"
val_file = f"in_vivo_0{FOLD}_test.xyz"
model_name = f"GN{FOLD}_official.pt"
plot_name = f"mse_GN{FOLD}_official.png"
time_file = "training_times_official.csv"

cutoff = 7.0
int_cutoff = cutoff
triplets_only = True
batch_size = 16
epochs = 50
learning_rate = 1e-3


class ASEAtomsDataContainer(DataContainer):
    """DataContainer-compatible wrapper for ASE Atoms lists.

    The important part is that batching/index construction is inherited from
    the official GemNet DataContainer instead of being hand-written here.
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


def run_epoch(dataset, model, optimizer, loss_fn, device, train):
    model.train() if train else model.eval()

    idxs = list(range(len(dataset)))
    if train:
        random.shuffle(idxs)

    total_loss = 0.0
    n_samples = 0

    # GemNet with direct_forces=False needs gradients in eval too, because
    # forward computes forces as -dE/dR.
    with torch.set_grad_enabled(True):
        for start_idx in range(0, len(idxs), batch_size):
            batch_ids = idxs[start_idx:start_idx + batch_size]
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


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print(f"cutoff = {cutoff}")
print(f"batch_size = {batch_size}")

if not os.path.exists("scale.json"):
    with open("scale.json", "w") as f:
        f.write("{}")

if not os.path.exists(time_file):
    with open(time_file, "w") as f:
        f.write(
            "Model;Fold;"
            "Total_seconds;Total_minutes;"
            "Train_total_s;Val_total_s;"
            "Avg_train_epoch_s;Avg_val_epoch_s\n"
        )

print(f"Fold {FOLD}: nacitavam data")
train_struct = read(train_file, index=":")
val_struct = read(val_file, index=":")
print(f"Train: {len(train_struct)} | Val: {len(val_struct)}")

print("Vytvaram official GemNet DataContainer wrappers")
train_data = ASEAtomsDataContainer(
    train_struct,
    cutoff=cutoff,
    int_cutoff=int_cutoff,
    triplets_only=triplets_only,
)
val_data = ASEAtomsDataContainer(
    val_struct,
    cutoff=cutoff,
    int_cutoff=int_cutoff,
    triplets_only=triplets_only,
)
del train_struct, val_struct
gc.collect()

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

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

best_val = float("inf")
train_times = []
val_times = []
train_losses = []
val_losses = []

print(f"Trenujem model {model_name}")
start = time.time()

for epoch in range(epochs):
    t0 = time.time()
    train_loss = run_epoch(train_data, model, optimizer, loss_fn, device, train=True)
    train_times.append(time.time() - t0)

    t0 = time.time()
    val_loss = run_epoch(val_data, model, optimizer, loss_fn, device, train=False)
    val_times.append(time.time() - t0)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), model_name)

    print(
        f"Epoch {epoch + 1:3d}/{epochs} | "
        f"Train MSE: {train_loss:.6f} | "
        f"Val MSE: {val_loss:.6f}"
    )

elapsed = time.time() - start
train_total = sum(train_times)
val_total = sum(val_times)
avg_train = train_total / len(train_times) if train_times else 0.0
avg_val = val_total / len(val_times) if val_times else 0.0

with open(time_file, "a") as f:
    f.write(
        f"GemNet-official;Fold {FOLD};"
        f"{elapsed:.2f};{elapsed / 60:.2f};"
        f"{train_total:.2f};{val_total:.2f};"
        f"{avg_train:.4f};{avg_val:.4f}\n"
    )


plt.figure(figsize=(7, 4.5))
plt.plot(range(1, epochs + 1), train_losses, label="Train MSE", marker="o", markersize=3)
plt.plot(range(1, epochs + 1), val_losses, label="Val MSE", marker="s", markersize=3)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title(f"GemNet official indexing - Fold {FOLD}")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(plot_name, dpi=300)
plt.close()

print(f"Fold {FOLD} hotovy")
print(f"Model ulozeny ako {model_name}")
print(f"Graf ulozeny ako {plot_name}")
print(f"Celkovy cas: {elapsed:.1f} s ({elapsed / 60:.2f} min)")
