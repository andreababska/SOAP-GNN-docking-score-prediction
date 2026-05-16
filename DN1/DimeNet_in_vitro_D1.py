import os
import time
import torch
import numpy as np

from ase.io import read
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNetPlusPlus


# =========================================================
# PARAMETER
# =========================================================

FOLD = 1

model_name  = f"DN{FOLD}.pt"
data_file   = "../../in_vitro.xyz"

output_file = f"in_vitro_DN{FOLD}.csv"
time_file   = "in_vitro_test_times.csv"

batch_size = 32


# =========================================================
# 1. DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Používam device:", device)


# =========================================================
# 2. LOAD MODEL
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

model.load_state_dict(torch.load(model_name, map_location=device))
model.eval()

print(f"Model {model_name} načítaný")


# =========================================================
# 3. LOAD in_vitro DATA
# =========================================================

structures = read(data_file, index=":")
print(f"Počet molekúl v in_vitro: {len(structures)}")


# =========================================================
# 4. CREATE PyG DATA + TIMING
# =========================================================

total_start = time.time()

print("Vytváram PyG dáta...")
data_start = time.time()

data_list = []
for s in structures:
    z = torch.tensor(s.get_atomic_numbers(), dtype=torch.long)
    pos = torch.tensor(s.get_positions(), dtype=torch.float)
    data_list.append(Data(z=z, pos=pos))

loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

data_time = time.time() - data_start
print(f"PyG dáta vytvorené za {data_time:.1f} s")


# =========================================================
# 5. PREDICTION + TIMING
# =========================================================

print("Robím predikciu...")
pred_start = time.time()

predictions = []

with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        y = model(batch.z, batch.pos, batch.batch)
        predictions.append(y.cpu().numpy().ravel())

predictions = np.concatenate(predictions)

pred_time = time.time() - pred_start
print(f"Predikcia trvala {pred_time:.1f} s")


# =========================================================
# 6. SAVE CSV + TIMING
# =========================================================

save_start = time.time()

with open(output_file, "w") as f:
    f.write("name;Predicted\n")
    for s, y in zip(structures, predictions):
        name = s.info.get("name", "unknown")
        f.write(f"{name};{y}\n")

save_time  = time.time() - save_start
total_time = time.time() - total_start

print(f"Výsledky uložené do {output_file}")
print(f"Uloženie CSV trvalo {save_time:.1f} s")


# =========================================================
# 7. SAVE TEST TIME
# =========================================================

if not os.path.exists(time_file):
    with open(time_file, "w") as f:
        f.write(
            "Model;Fold;"
            "Total_seconds;Total_minutes;"
            "Data_seconds;Prediction_seconds;Save_seconds;"
            "Num_molecules\n"
        )

with open(time_file, "a") as f:
    f.write(
        f"DimeNet;Fold {FOLD};"
        f"{total_time:.2f};{total_time/60:.2f};"
        f"{data_time:.2f};{pred_time:.2f};{save_time:.2f};"
        f"{len(structures)}\n"
    )

print(f"Čas uložený do {time_file}")


# =========================================================
# 8. SUMMARY
# =========================================================

print("\nHotovo")
print(f"Model: {model_name}")
print(f"Dáta: {data_file}")
print(f"Počet molekúl: {len(structures)}")
print(f"Celkový čas: {total_time:.1f} s ({total_time/60:.2f} min)")
print(f"Data čas: {data_time:.1f} s")
print(f"Predikcia čas: {pred_time:.1f} s")