import os
import time
import numpy as np
import tensorflow as tf

from ase.io import read
from dscribe.descriptors import SOAP

import matplotlib
matplotlib.use("Agg")  # bezpecne pre headless cluster
import matplotlib.pyplot as plt


# =========================================================
# PARAMETER
# =========================================================

FOLD = 1

train_file = f"in_vivo_0{FOLD}_train.xyz"
val_file   = f"in_vivo_0{FOLD}_test.xyz"     # held-out fold v 5-fold CV
model_name = f"SOAP{FOLD}.keras"
plot_name  = f"loss_SOAP{FOLD}.png"
time_file  = "training_times.csv"


# =========================================================
# 1. LOAD TRAIN + VAL DATA
# =========================================================

print(f"Fold {FOLD}: načítavam dáta")
train_struct = read(train_file, index=":")
val_struct   = read(val_file, index=":")
print(f"Train: {len(train_struct)} | Val (held-out fold): {len(val_struct)}")


# =========================================================
# 2. SOAP DESCRIPTOR
# =========================================================

# Zbierame species zo vsetkych struktur (train + val), aby val nemohol obsahovat
# atom, ktory by SOAP neocakaval.
species = set()
for s in train_struct:
    species.update(s.get_chemical_symbols())
for s in val_struct:
    species.update(s.get_chemical_symbols())

soap = SOAP(
    species=species,
    r_cut=17.0,
    n_max=7,
    l_max=7,
    sigma=0.5,
    average="outer",
    sparse=False,
    periodic=False
)


# =========================================================
# 3. CREATE SOAP FEATURES
# =========================================================

print("Počítam SOAP deskriptory...")
soap_start = time.time()

x_train = soap.create(train_struct, n_jobs=-1)
x_val   = soap.create(val_struct,   n_jobs=-1)

y_train = np.array([s.get_potential_energy() for s in train_struct])
y_val   = np.array([s.get_potential_energy() for s in val_struct])

print("SOAP dimenzia:", x_train.shape[1])
print(f"SOAP výpočet trval {time.time() - soap_start:.1f} s")


# =========================================================
# 4. NEURAL NETWORK
# =========================================================

def build_nn(input_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
        70,
        activation="relu",
        kernel_initializer="random_uniform",
    ))
    model.add(tf.keras.layers.Dense(25, activation="relu"))
    model.add(tf.keras.layers.Dense(15, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="linear"))
    return model


model = build_nn(x_train.shape[1])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mean_squared_error",
    metrics=["mean_squared_error"]
)


# =========================================================
# 5. TIMING CALLBACK
# =========================================================
# Keras fit flow v kazdej epoche:
#   on_epoch_begin  -> [train batches] -> on_test_begin -> [val batches] -> on_test_end -> on_epoch_end
# Meriame train fazu od epoch_begin po test_begin a val fazu od test_begin po test_end.

class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_times = []
        self.val_times = []
        self._train_start = None
        self._val_start = None

    def on_epoch_begin(self, epoch, logs=None):
        self._train_start = time.time()

    def on_test_begin(self, logs=None):
        # zaciatok validacie = koniec trenovacej fazy
        self._val_start = time.time()
        if self._train_start is not None:
            self.train_times.append(self._val_start - self._train_start)

    def on_test_end(self, logs=None):
        if self._val_start is not None:
            self.val_times.append(time.time() - self._val_start)


# =========================================================
# 6. TRAINING
# =========================================================

epochs = 50

if not os.path.exists(time_file):
    with open(time_file, "w") as f:
        f.write(
            "Model;Fold;"
            "Total_seconds;Total_minutes;"
            "Train_total_s;Val_total_s;"
            "Avg_train_epoch_s;Avg_val_epoch_s\n"
        )

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_name,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    mode="min"
)
timer_cb = EpochTimer()

print(f"Trénujem SOAP model → {model_name}")
start = time.time()

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=32,
    verbose=1,
    callbacks=[checkpoint_cb, timer_cb]
)

end = time.time()
elapsed = end - start

train_total = sum(timer_cb.train_times)
val_total   = sum(timer_cb.val_times)
avg_train   = train_total / len(timer_cb.train_times) if timer_cb.train_times else 0.0
avg_val     = val_total   / len(timer_cb.val_times)   if timer_cb.val_times   else 0.0

print(f"\nFold {FOLD} hotový")
print(f"Model uložený ako {model_name}")
print(f"Celkový čas: {elapsed:.1f} s ({elapsed/60:.2f} min)")
print(f"Train total: {train_total:.1f} s | Val total: {val_total:.1f} s")
print(f"Priemer/epocha — train: {avg_train:.2f} s | val: {avg_val:.2f} s")


# =========================================================
# 7. SAVE TRAINING TIME
# =========================================================

with open(time_file, "a") as f:
    f.write(
        f"SOAP;Fold {FOLD};"
        f"{elapsed:.2f};{elapsed/60:.2f};"
        f"{train_total:.2f};{val_total:.2f};"
        f"{avg_train:.4f};{avg_val:.4f}\n"
    )

print(f"Čas uložený do {time_file}")


# =========================================================
# 8. PLOT LOSS FUNCTION (Train + Val MSE vs Epochs)
# =========================================================

train_losses = history.history["loss"]
val_losses   = history.history["val_loss"]
n_epochs = len(train_losses)

plt.figure(figsize=(7, 4.5))

plt.plot(
    range(1, n_epochs + 1),
    train_losses,
    label="Train MSE",
    marker="o",
    markersize=3,
    linewidth=1.5
)

plt.plot(
    range(1, n_epochs + 1),
    val_losses,
    label="Val MSE",
    marker="s",
    markersize=3,
    linewidth=1.5
)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title(f"Loss function — SOAP, Fold {FOLD}")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(plot_name, dpi=300)
plt.close()

print(f"Graf uložený ako {plot_name}")

tf.keras.backend.clear_session()