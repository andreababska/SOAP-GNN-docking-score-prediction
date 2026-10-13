import os
import gc
import time
import numpy as np
import tensorflow as tf

from ase.io import read
from dscribe.descriptors import SOAP


# =========================================================
# PARAMETER
# =========================================================

FOLD = 1

model_name = f"SOAP{FOLD}.keras"
data_file = "../../in_vitro.xyz"

output_file = f"in_vitro_SOAP{FOLD}.csv"
time_file = "in_vitro_test_times.csv"

chunk_size = 10000
predict_batch_size = 32


# =========================================================
# 1. LOAD MODEL
# =========================================================

model = tf.keras.models.load_model(model_name)
print(f"Model {model_name} načítaný")


# =========================================================
# 2. LOAD in_vitro DATA
# =========================================================

structures = read(data_file, index=":")
print(f"Počet molekúl v in_vitro: {len(structures)}")


# =========================================================
# 3. SOAP DESCRIPTOR
# =========================================================

species = set()
for s in structures:
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

print("Pocitam SOAP deskriptory po chunkoch...")
print(f"chunk_size = {chunk_size}")
print(f"predict_batch_size = {predict_batch_size}")

total_start = time.time()
soap_time = 0.0
pred_time = 0.0
save_time = 0.0

with open(output_file, "w") as f:
    f.write("name;Predicted\n")

    n = len(structures)
    n_chunks = (n + chunk_size - 1) // chunk_size

    for chunk_idx, start_idx in enumerate(range(0, n, chunk_size), start=1):
        chunk_structures = structures[start_idx:start_idx + chunk_size]

        soap_start = time.time()
        x_chunk = soap.create(chunk_structures, n_jobs=-1)
        soap_time += time.time() - soap_start

        pred_start = time.time()
        y_chunk = model.predict(
            x_chunk,
            batch_size=predict_batch_size,
            verbose=0,
        ).reshape(-1)
        pred_time += time.time() - pred_start

        save_start = time.time()
        for s, y in zip(chunk_structures, y_chunk):
            name = s.info.get("name", "unknown")
            f.write(f"{name};{y}\n")
        f.flush()
        save_time += time.time() - save_start

        done = min(start_idx + chunk_size, n)
        print(
            f"  chunk {chunk_idx}/{n_chunks}: "
            f"{len(chunk_structures)} molekul, hotovo {done}/{n}"
        )

        del chunk_structures, x_chunk, y_chunk
        gc.collect()

total_time = time.time() - total_start

print(f"Vysledky ulozene do {output_file}")
print(f"SOAP vypocet trval {soap_time:.1f} s")
print(f"Predikcia trvala {pred_time:.1f} s")
print(f"Ulozenie CSV trvalo {save_time:.1f} s")

if not os.path.exists(time_file):
    with open(time_file, "w") as f:
        f.write(
            "Model;Fold;"
            "Total_seconds;Total_minutes;"
            "SOAP_seconds;Prediction_seconds;Save_seconds;"
            "Num_molecules\n"
        )

with open(time_file, "a") as f:
    f.write(
        f"SOAP;Fold {FOLD};"
        f"{total_time:.2f};{total_time/60:.2f};"
        f"{soap_time:.2f};{pred_time:.2f};{save_time:.2f};"
        f"{len(structures)}\n"
    )

print(f"Cas ulozeny do {time_file}")

print("\nHotovo")
print(f"Model: {model_name}")
print(f"Data: {data_file}")
print(f"Pocet molekul: {len(structures)}")
print(f"Celkovy cas: {total_time:.1f} s ({total_time/60:.2f} min)")
print(f"SOAP cas: {soap_time:.1f} s")
print(f"Predikcia cas: {pred_time:.1f} s")

tf.keras.backend.clear_session()
raise SystemExit(0)


# =========================================================
# 4. CREATE SOAP FEATURES + TIMING
# =========================================================

print("Počítam SOAP deskriptory...")
total_start = time.time()

soap_start = time.time()
x = soap.create(structures, n_jobs=-1)
soap_time = time.time() - soap_start

print("SOAP shape:", x.shape)
print(f"SOAP výpočet trval {soap_time:.1f} s")


# =========================================================
# 5. PREDICTION + TIMING
# =========================================================

print("Robím predikciu...")

pred_start = time.time()
y_pred = model.predict(x, batch_size=32).reshape(-1)
pred_time = time.time() - pred_start

print(f"Predikcia trvala {pred_time:.1f} s")


# =========================================================
# 6. SAVE CSV
# =========================================================

save_start = time.time()

with open(output_file, "w") as f:
    f.write("name;Predicted\n")
    for s, y in zip(structures, y_pred):
        name = s.info.get("name", "unknown")
        f.write(f"{name};{y}\n")

save_time = time.time() - save_start
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
            "SOAP_seconds;Prediction_seconds;Save_seconds;"
            "Num_molecules\n"
        )

with open(time_file, "a") as f:
    f.write(
        f"SOAP;Fold {FOLD};"
        f"{total_time:.2f};{total_time/60:.2f};"
        f"{soap_time:.2f};{pred_time:.2f};{save_time:.2f};"
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
print(f"SOAP čas: {soap_time:.1f} s")
print(f"Predikcia čas: {pred_time:.1f} s")

tf.keras.backend.clear_session()
