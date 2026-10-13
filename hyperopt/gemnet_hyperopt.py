# =========================================================
# LIBRARIES
# =========================================================

import gc, os
os.environ.pop("NETWORKX_BACKENDS", None)
os.environ.pop("NETWORKX_BACKEND", None)

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from ase.io import read
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ocpmodels.models.gemnet.gemnet import GemNet


# =========================================================
# LOAD DATA
# =========================================================

file_name = "../../in_vivo.xyz"
print(f"Loading data: {file_name}")

structures = read(file_name, index=":5000")  # menší subset (GemNet je heavy)
print(f"Structures: {len(structures)}")

targets = [float(s.get_potential_energy()) for s in structures]

data_list = []
for mol, y_val in zip(structures, targets):
    z = torch.tensor(mol.get_atomic_numbers(), dtype=torch.long)
    pos = torch.tensor(mol.get_positions(), dtype=torch.float)
    y = torch.tensor([[y_val]], dtype=torch.float)

    data_list.append(Data(z=z, pos=pos, y=y))

print(f"Dataset size: {len(data_list)}")

# =========================================================
# SPLIT
# =========================================================

train_data, val_data = train_test_split(
    data_list, test_size=0.2, random_state=42
)

print(f"Train: {len(train_data)} | Val: {len(val_data)}")

# =========================================================
# DEVICE
# =========================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# =========================================================
# OBJECTIVE FUNCTION
# =========================================================

def objective(params):
    try:
        print("Params:", params)

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader   = DataLoader(val_data, batch_size=16, shuffle=False)

        model = GemNet(
            num_atoms=None,
            bond_feat_dim=0,
            num_targets=1,

            num_blocks=int(params["num_blocks"]),
            num_radial=int(params["num_radial"]),
            num_spherical=int(params["num_spherical"]),

            emb_size_atom=params["emb_size_atom"],
            emb_size_edge=params["emb_size_edge"],
            emb_size_trip=params["emb_size_trip"],

            cutoff=params["cutoff"],
            max_neighbors=int(params["max_neighbors"])
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()

        best_val = float("inf")

        for epoch in range(30):  # menej epoch (GemNet je heavy)
            model.train()

            for batch in train_loader:
                batch = batch.to(device)

                optimizer.zero_grad()
                pred = model(z=batch.z, pos=batch.pos, batch=batch.batch)
                loss = loss_fn(pred, batch.y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(z=batch.z, pos=batch.pos, batch=batch.batch)
                    val_loss += loss_fn(pred, batch.y).item()

            val_loss /= len(val_loader)

            if val_loss < best_val:
                best_val = val_loss

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"MSE: {best_val}")
        return {"loss": best_val, "status": STATUS_OK}

    except Exception as e:
        print("Error:", e)
        return {"loss": 1e6, "status": STATUS_OK}


# =========================================================
# HYPERPARAMETER SPACE
# =========================================================

space = {
    "num_blocks": hp.quniform("num_blocks", 2, 4, 1),
    "num_radial": hp.quniform("num_radial", 4, 8, 1),
    "num_spherical": hp.quniform("num_spherical", 4, 8, 1),

    "emb_size_atom": hp.choice("emb_size_atom", [64, 128]),
    "emb_size_edge": hp.choice("emb_size_edge", [64, 128]),
    "emb_size_trip": hp.choice("emb_size_trip", [32, 64]),

    "cutoff": hp.uniform("cutoff", 5.0, 8.0),
    "max_neighbors": hp.quniform("max_neighbors", 20, 40, 1),
}

# =========================================================
# RUN OPTIMIZATION
# =========================================================

trials = Trials()

best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,   # začni menšie
    trials=trials,
    rstate=np.random.default_rng(42)
)

print("Best params:", best)

# =========================================================
# EXPORT RESULTS
# =========================================================

def export_trials_to_csv(trials, filename="GemNet_hyperopt.csv"):
    records = []

    emb_atom_vals = [64, 128]
    emb_edge_vals = [64, 128]
    emb_trip_vals = [32, 64]

    for trial in trials.trials:
        vals = trial["misc"]["vals"]

        row = {
            "num_blocks": int(vals["num_blocks"][0]),
            "num_radial": int(vals["num_radial"][0]),
            "num_spherical": int(vals["num_spherical"][0]),

            "emb_size_atom": emb_atom_vals[vals["emb_size_atom"][0]],
            "emb_size_edge": emb_edge_vals[vals["emb_size_edge"][0]],
            "emb_size_trip": emb_trip_vals[vals["emb_size_trip"][0]],

            "cutoff": vals["cutoff"][0],
            "max_neighbors": int(vals["max_neighbors"][0]),

            "mse": trial["result"]["loss"]
        }

        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(filename, sep=";", index=False)
    print(f"Saved: {filename}")

export_trials_to_csv(trials)