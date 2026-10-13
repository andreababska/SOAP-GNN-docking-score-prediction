import os
import torch
import time
from ase.io import read

import matplotlib
matplotlib.use("Agg")  # bezpecne pre headless cluster
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNetPlusPlus


# =========================================================
# PARAMETER
# =========================================================

FOLD = 1

train_file = f"in_vivo_0{FOLD}_train.xyz"
val_file   = f"in_vivo_0{FOLD}_test.xyz"     # held-out fold v 5-fold CV
model_name = f"DN{FOLD}.pt"
plot_name  = f"loss_DN{FOLD}.png"
time_file  = "training_times.csv"


# =========================================================
# 1. LOAD TRAIN + VAL DATA
# =========================================================

print(f"Fold {FOLD}: načítavam dáta")
train_struct = read(train_file, index=":")
val_struct   = read(val_file, index=":")
print(f"Train: {len(train_struct)} | Val (held-out fold): {len(val_struct)}")


# =========================================================
# 2. ASE → PyG
# =========================================================

def ase_to_pyg(structures):
    data_list = []
    for mol in structures:
        z = torch.tensor(mol.get_atomic_numbers(), dtype=torch.long)
        pos = torch.tensor(mol.get_positions(), dtype=torch.float)
        y = torch.tensor([[float(mol.get_potential_energy())]], dtype=torch.float)
        data_list.append(Data(z=z, pos=pos, y=y))
    return data_list


train_data = ase_to_pyg(train_struct)
val_data   = ase_to_pyg(val_struct)


# =========================================================
# 3. DEVICE + DATALOADER
# =========================================================

batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if not os.path.exists(time_file):
    with open(time_file, "w") as f:
        f.write(
            "Model;Fold;"
            "Total_seconds;Total_minutes;"
            "Train_total_s;Val_total_s;"
            "Avg_train_epoch_s;Avg_val_epoch_s\n"
        )

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)


# =========================================================
# 4. MODEL
# =========================================================

model = DimeNetPlusPlus(
    hidden_channels=128,
    out_channels=1,
    num_blocks=3,
    num_radial=3,
    num_spherical=12,
    int_emb_size=128,
    basis_emb_size=10,
    out_emb_channels=32,
    cutoff=6.5,
    max_num_neighbors=39
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()


# =========================================================
# 5. TRAINING
# =========================================================

epochs = 50
best_val = float("inf")

train_times  = []
val_times    = []
train_losses = []
val_losses   = []

print(f"Trénujem model {model_name}")
start = time.time()

for epoch in range(epochs):
    # --- TRAIN ---
    train_epoch_start = time.time()
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.z, batch.pos, batch.batch)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_times.append(time.time() - train_epoch_start)

    # --- VAL ---
    val_epoch_start = time.time()
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch.z, batch.pos, batch.batch)
            val_loss += loss_fn(pred, batch.y).item()
    val_loss /= len(val_loader)
    val_times.append(time.time() - val_epoch_start)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), model_name)

    print(
        f"Epoch {epoch+1:3d}/{epochs} | "
        f"Train MSE: {train_loss:.6f} | "
        f"Val MSE: {val_loss:.6f}"
    )

end = time.time()
elapsed = end - start

train_total = sum(train_times)
val_total   = sum(val_times)
avg_train   = train_total / len(train_times) if train_times else 0.0
avg_val     = val_total   / len(val_times)   if val_times   else 0.0

print(f"\nFold {FOLD} hotový")
print(f"Model uložený ako {model_name}")
print(f"Celkový čas: {elapsed:.1f} s ({elapsed/60:.2f} min)")
print(f"Train total: {train_total:.1f} s | Val total: {val_total:.1f} s")
print(f"Priemer/epocha — train: {avg_train:.2f} s | val: {avg_val:.2f} s")


# =========================================================
# 6. SAVE TRAINING TIME
# =========================================================

with open(time_file, "a") as f:
    f.write(
        f"DimeNet;Fold {FOLD};"
        f"{elapsed:.2f};{elapsed/60:.2f};"
        f"{train_total:.2f};{val_total:.2f};"
        f"{avg_train:.4f};{avg_val:.4f}\n"
    )

print(f"Čas uložený do {time_file}")


# =========================================================
# 7. PLOT LOSS FUNCTION (Train + Val MSE vs Epochs)
# =========================================================

epochs_range = range(1, epochs + 1)

plt.figure(figsize=(7, 4.5))

plt.plot(
    epochs_range,
    train_losses,
    label="Train MSE",
    marker="o",
    markersize=3,
    linewidth=1.5
)

plt.plot(
    epochs_range,
    val_losses,
    label="Val MSE",
    marker="s",
    markersize=3,
    linewidth=1.5
)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title(f"Loss function — DimeNet++, Fold {FOLD}")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig(plot_name, dpi=300)
plt.close()

print(f"Graf uložený ako {plot_name}")