# === KNIŽNICE ===
import gc
import os
import numpy as np
import pandas as pd
import torch

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope                       

from ase.io import read
from sklearn.model_selection import train_test_split

from torch_geometric.nn import DimeNetPlusPlus
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Odstránenie konfliktných premenných prostredia pre NetworkX
os.environ.pop("NETWORKX_BACKENDS", None)
os.environ.pop("NETWORKX_BACKEND", None)


# ==============================================================================
# === NAČÍTANIE DÁT ===
# ==============================================================================

file_name = "../../in_vivo.xyz"
print(f"Načítavam dáta zo súboru: {file_name}")

structures = read(file_name, index=":10000")
print(f"Počet molekúl v datasete: {len(structures)}")

# Získanie cieľovej premennej – DS energie z každej molekuly
chem_property = [float(s.get_potential_energy()) for s in structures]

data_list = []
for mol, DS in zip(structures, chem_property):
    z   = torch.tensor(mol.get_atomic_numbers(), dtype=torch.long)
    pos = torch.tensor(mol.get_positions(),       dtype=torch.float)
    y   = torch.tensor([[float(DS)]],             dtype=torch.float)
    data_list.append(Data(z=z, pos=pos, y=y))

print(f"Načítaných molekúl: {len(data_list)}")

# Rozdelenie: 80 % tréning / 20 % validácia
train_data, val_data = train_test_split(data_list, test_size=0.20, random_state=42)
print(f"Train: {len(train_data)} | Val: {len(val_data)}")


# ==============================================================================
# === DETEKCIA ZARIADENIA ===
# ==============================================================================

if torch.cuda.is_available():
    device      = torch.device("cuda")
    pin_memory  = True
    num_workers = 0
    print("Používam GPU:", torch.cuda.get_device_name(0))
else:
    device      = torch.device("cpu")
    pin_memory  = False
    num_workers = 0
    print("Používam CPU")


# ==============================================================================
# === POMOCNÁ FUNKCIA – early stopping ===
# ==============================================================================

class EarlyStopper:
    """Zastaví tréning ak sa validačná strata nezlepší počas `patience` epoch."""
    def __init__(self, patience=10):
        self.patience   = patience
        self.counter    = 0
        self.best_val   = float("inf")

    def step(self, val_loss):
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.counter  = 0
            return False        # pokračuj
        self.counter += 1
        return self.counter >= self.patience  # zastav


# ==============================================================================
# === CIEĽOVÁ FUNKCIA PRE HYPEROPT ===
# ==============================================================================

BEST_GLOBAL_VAL = float("inf")   # sledovanie globálne najlepšieho modelu


def objective(params):
    global BEST_GLOBAL_VAL

    try:
        print(f"\nTestujem parametre: {params}")

        batch_size = params['batch_size']

        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            pin_memory=pin_memory, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False,
            pin_memory=pin_memory, num_workers=num_workers
        )

        # --- Model ---
        model = DimeNetPlusPlus(
            hidden_channels  = params['hidden_channels'],
            out_channels     = 1,
            num_blocks       = params['num_blocks'],
            num_radial       = params['num_radial'],
            num_spherical    = params['num_spherical'],
            int_emb_size     = params['int_emb_size'],
            basis_emb_size   = params['basis_emb_size'],
            out_emb_channels = params['out_emb_channels'],
            cutoff           = params['cutoff'],
            max_num_neighbors= params['max_num_neighbors'],
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr           = params['lr'],
            weight_decay = params['weight_decay']
        )

        # Learning rate scheduler – zníži LR ak sa validácia zlepší
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        loss_fn      = torch.nn.MSELoss()
        early_stop   = EarlyStopper(patience=10)
        best_val     = float("inf")

        epochs = 50
        for epoch in range(epochs):

            # --- Tréning ---
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                preds = model(batch.z, batch.pos, batch.batch)
                loss  = loss_fn(preds, batch.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # --- Validácia ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    preds    = model(batch.z, batch.pos, batch.batch)
                    val_loss += loss_fn(preds, batch.y).item()
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss

                # Ulož globálne najlepší model
                if best_val < BEST_GLOBAL_VAL:
                    BEST_GLOBAL_VAL = best_val
                    torch.save(model.state_dict(), "best_dimenet_model.pt")
                    print(f"  ✓ Nový globálny najlepší model uložený (val MSE={best_val:.6f})")

            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | "
                  f"Best Val: {best_val:.6f}")

            if early_stop.step(val_loss):
                print(f"  Early stopping po {epoch+1} epochách.")
                break

        mse = best_val

    except Exception as e:
        print(f"  Chyba pri tréningu: {e}")
        mse = 1e6

    finally:
        # Uvoľnenie pamäte po každom pokuse
        try:
            del model, optimizer, scheduler
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  → Výsledok pokusu – Val MSE: {mse:.6f}")
    return {'loss': mse, 'status': STATUS_OK}


# ==============================================================================
# === HYPERPARAMETROVÝ PRIESTOR ===
# ==============================================================================

space = {
    'lr':               hp.loguniform('lr',            np.log(1e-4), np.log(1e-2)),
    'weight_decay':     hp.loguniform('weight_decay',  np.log(1e-6), np.log(1e-3)),
    'batch_size':       hp.choice('batch_size',        [16, 32, 64]),

    'hidden_channels':  hp.choice('hidden_channels',   [64, 128, 256]),
    'num_blocks':       scope.int(hp.quniform('num_blocks',    3, 6, 1)),
    'num_radial':       scope.int(hp.quniform('num_radial',    4, 12, 2)),
    'num_spherical':    scope.int(hp.quniform('num_spherical', 4, 8,  1)),
    'int_emb_size':     hp.choice('int_emb_size',      [32, 64, 128]),
    'basis_emb_size':   scope.int(hp.quniform('basis_emb_size', 4, 16, 4)),
    'out_emb_channels': hp.choice('out_emb_channels',  [64, 128, 256]),

    'cutoff':           hp.uniform('cutoff',           4.0, 8.0),
    'max_num_neighbors':scope.int(hp.quniform('max_num_neighbors', 16, 64, 8)),
}


# ==============================================================================
# === SPUSTENIE OPTIMALIZÁCIE ===
# ==============================================================================

trials = Trials()

best = fmin(
    fn      = objective,
    space   = space,
    algo    = tpe.suggest,
    max_evals = 50,
    trials  = trials,
    rstate  = np.random.default_rng(42)
)

print("\n=== Najlepšie nájdené parametre ===")
print(best)


# ==============================================================================
# === EXPORT VÝSLEDKOV DO CSV ===
# ==============================================================================

def export_trials_to_csv(trials, filename="DimeNet_hyperopt.csv"):
    """
    Exportuje všetky pokusy Hyperopt do CSV.
    hp.choice vracia index → mapujeme späť na skutočné hodnoty.
    """
    batch_size_vals       = [16, 32, 64]
    hidden_channels_vals  = [64, 128, 256]
    int_emb_size_vals     = [32, 64, 128]
    out_emb_channels_vals = [64, 128, 256]  

    records = []
    for trial in trials.trials:
        vals = trial['misc']['vals']
        row = {
            'lr':               vals['lr'][0],
            'weight_decay':     vals['weight_decay'][0],
            'batch_size':       batch_size_vals[vals['batch_size'][0]],
            'hidden_channels':  hidden_channels_vals[vals['hidden_channels'][0]],
            'num_blocks':       int(vals['num_blocks'][0]),
            'num_radial':       int(vals['num_radial'][0]),
            'num_spherical':    int(vals['num_spherical'][0]),
            'int_emb_size':     int_emb_size_vals[vals['int_emb_size'][0]],
            'basis_emb_size':   int(vals['basis_emb_size'][0]),
            'out_emb_channels': out_emb_channels_vals[vals['out_emb_channels'][0]],
            'cutoff':           vals['cutoff'][0],
            'max_num_neighbors':int(vals['max_num_neighbors'][0]),
            'val_mse':          trial['result']['loss'],
        }
        records.append(row)

    df = pd.DataFrame(records).sort_values('val_mse')
    df.to_csv(filename, sep=";", index=False)
    print(f"\nVýsledky exportované do: {filename}")
    print(df.head(10).to_string(index=False))


export_trials_to_csv(trials)
