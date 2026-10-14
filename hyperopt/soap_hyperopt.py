# === KNIŽNICE ===
import gc
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from dscribe.descriptors import SOAP
from ase.io import read
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# === NAČÍTANIE DÁT ===
# ==============================================================================

file_name = "../in_vivo.xyz"
print(f"Načítavam dáta zo súboru: {file_name}")

structures = read(file_name, index=":10000")
print(f"Počet molekúl v datasete: {len(structures)}")

# Cieľová premenná – potenciálna energia každej molekuly
chem_property = [float(s.get_potential_energy()) for s in structures]

# Všetky atómové druhy prítomné v datasete
species = sorted(set(atom for s in structures for atom in s.get_chemical_symbols()))
print(f"Atómové druhy: {species}")

# Rozdelenie: 80 % tréning / 20 % validácia
structures_train, structures_val, y_train, y_val = train_test_split(
    structures, chem_property, test_size=0.20, random_state=42
)

print(f"Train: {len(structures_train)} | Val: {len(structures_val)}")

y_train = np.asarray(y_train, dtype=np.float32)
y_val   = np.asarray(y_val,   dtype=np.float32)


# ==============================================================================
# === POMOCNÉ FUNKCIE ===
# ==============================================================================

def build_model(size_input, layer_sizes, activation):
    """
    Zostaví sekvenčný Keras model s premenlivým počtom skrytých vrstiev.
    layer_sizes: zoznam počtov neurónov pre každú skrytú vrstvu.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(size_input,)))
    for units in layer_sizes:
        model.add(tf.keras.layers.Dense(units, activation=activation))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    return model


def get_optimizer(name, lr):
    """Vráti Keras optimizer podľa mena a learning rate."""
    if name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    raise ValueError(f"Neznámy optimizer: {name}")


# ==============================================================================
# === CIEĽOVÁ FUNKCIA PRE HYPEROPT ===
# ==============================================================================

BEST_GLOBAL_MSE = float("inf")


def objective(params):
    global BEST_GLOBAL_MSE

    try:
        print(f"\nTestujem parametre: {params}")

        # --- l_max nesmie byť väčší ako n_max ---
        n_max = int(params['n_max'])
        l_max = min(int(params['l_max']), n_max)

        # --- SOAP deskriptor ---
        soap = SOAP(
            species  = species,
            r_cut    = params['r_cut'],
            n_max    = n_max,
            l_max    = l_max,
            sigma    = params['sigma'],
            average  = 'outer',
            periodic = False,
            sparse   = False
        )

        n_jobs = max(1, os.cpu_count())
        x_train_raw = np.asarray(soap.create(structures_train, n_jobs=n_jobs), dtype=np.float32)
        x_val_raw   = np.asarray(soap.create(structures_val,   n_jobs=n_jobs), dtype=np.float32)
        
        # --- Normalizácia SOAP príznakov (fit len na tréningových dátach) ---
        scaler  = StandardScaler()
        x_train = scaler.fit_transform(x_train_raw)
        x_val   = scaler.transform(x_val_raw)

        # --- Zostavenie vrstiev podľa n_layers ---
        n_layers    = int(params['n_layers'])
        all_sizes   = [
            int(params['n_nn1']),
            int(params['n_nn2']),
            int(params['n_nn3']),
            int(params['n_nn4']),
        ]
        layer_sizes = all_sizes[:n_layers]   # použijeme len prvých n_layers vrstiev

        # --- Model ---
        model = build_model(
            size_input  = x_train.shape[1],
            layer_sizes = layer_sizes,
            activation  = params['af']
        )

        opt = get_optimizer(params['opt'], params['lr'])
        model.compile(loss='mean_squared_error', optimizer=opt,
                      metrics=['mean_squared_error'])

        # --- Callbacks ---
        checkpoint_file = "soap_best_model.keras"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath        = checkpoint_file,
                monitor         = 'val_loss',
                save_best_only  = True,
                save_weights_only = False,
                verbose         = 0
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor   = 'val_loss',
                patience  = 15,           # zastav ak sa val_loss nezlepší 15 epoch
                restore_best_weights = True,
                verbose   = 0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor  = 'val_loss',
                factor   = 0.5,           # zníž LR na polovicu
                patience = 7,
                min_lr   = 1e-7,
                verbose  = 0
            ),
        ]

        # --- Tréning ---
        model.fit(
            x_train, y_train,
            epochs          = 200,        # viac epoch – early stopping zastaví skôr
            batch_size      = int(params['bs']),
            validation_data = (x_val, y_val),
            callbacks       = callbacks,
            verbose         = 0
        )

        # --- Vyhodnotenie na validačnej množine ---
        best_model = tf.keras.models.load_model(checkpoint_file)
        mse = float(best_model.evaluate(x_val, y_val, verbose=0)[0])

        # --- Uloženie globálne najlepšieho modelu ---
        if mse < BEST_GLOBAL_MSE:
            BEST_GLOBAL_MSE = mse
            best_model.save("soap_global_best_model.keras")
            print(f"  ✓ Nový globálny najlepší model uložený (val MSE={mse:.6f})")

    except Exception as e:
        print(f"  Chyba: {e}")
        mse = 1e6

    finally:
        tf.keras.backend.clear_session()
        gc.collect()

    print(f"  → Výsledok pokusu – Val MSE: {mse:.6f}")
    return {'loss': mse, 'status': STATUS_OK}


# ==============================================================================
# === HYPERPARAMETROVÝ PRIESTOR ===
# ==============================================================================
#

space = {
    # --- SOAP ---
    'r_cut':   hp.uniform('r_cut',  4.0, 10.0),
    'n_max':   scope.int(hp.quniform('n_max', 3, 9, 1)),
    'l_max':   scope.int(hp.quniform('l_max', 3, 9, 1)),   
    'sigma':   hp.uniform('sigma',  0.1,  1.5),

    # --- Optimizer ---
    'opt':     hp.choice('opt', ['adam', 'sgd']),
    'lr':      hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),

    # --- Architektúra siete ---
    'bs':      hp.choice('bs', [32, 64, 128]),
    'n_layers':hp.choice('n_layers', [2, 3, 4]),
    'n_nn1':   hp.choice('n_nn1', [32,  64,  128, 256]),
    'n_nn2':   hp.choice('n_nn2', [16,  32,   64, 128]),
    'n_nn3':   hp.choice('n_nn3', [8,   16,   32,  64]),
    'n_nn4':   hp.choice('n_nn4', [4,    8,   16,  32]),
    'af':      hp.choice('af', ['relu', 'softplus', 'tanh']),
}


# ==============================================================================
# === SPUSTENIE OPTIMALIZÁCIE ===
# ==============================================================================

trials = Trials()

best = fmin(
    fn        = objective,
    space     = space,
    algo      = tpe.suggest,
    max_evals = 50,
    trials    = trials,
    rstate    = np.random.default_rng(42)
)

print("\n=== Najlepšie nájdené parametre ===")
print(best)


# ==============================================================================
# === EXPORT VÝSLEDKOV DO CSV ===
# ==============================================================================

def export_trials_to_csv(trials, filename="SOAP_hyperopt.csv"):
    """
    Exportuje všetky pokusy do CSV.
    hp.choice vracia index → mapujeme späť na skutočné hodnoty.
    """
    opt_vals     = ['adam', 'sgd']
    bs_vals      = [32, 64, 128]
    n_layers_vals= [2, 3, 4]
    nn1_vals     = [32,  64,  128, 256]
    nn2_vals     = [16,  32,   64, 128]
    nn3_vals     = [8,   16,   32,  64]
    nn4_vals     = [4,    8,   16,  32]
    af_vals      = ['relu', 'softplus', 'tanh']

    records = []
    for trial in trials.trials:
        vals = trial['misc']['vals']

        n_max = int(vals['n_max'][0])
        l_max = min(int(vals['l_max'][0]), n_max)  

        row = {
            'r_cut':    vals['r_cut'][0],
            'n_max':    n_max,
            'l_max':    l_max,
            'sigma':    vals['sigma'][0],
            'opt':      opt_vals[vals['opt'][0]],
            'lr':       vals['lr'][0],
            'bs':       bs_vals[vals['bs'][0]],
            'n_layers': n_layers_vals[vals['n_layers'][0]],
            'n_nn1':    nn1_vals[vals['n_nn1'][0]],
            'n_nn2':    nn2_vals[vals['n_nn2'][0]],
            'n_nn3':    nn3_vals[vals['n_nn3'][0]],
            'n_nn4':    nn4_vals[vals['n_nn4'][0]],
            'af':       af_vals[vals['af'][0]],
            'test_mse': trial['result']['loss'],
        }
        records.append(row)

    df = pd.DataFrame(records).sort_values('test_mse')
    df.to_csv(filename, sep=";", index=False)
    print(f"\nVýsledky exportované do: {filename}")
    print(df.head(10).to_string(index=False))


export_trials_to_csv(trials)