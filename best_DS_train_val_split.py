from ase.io import read, write
import random

# =========================
# Nastavenia
# =========================
input_file = "in_vivo.xyz"
train_file = "train.xyz"
val_file = "val.xyz"

n_train = 10000
n_val = 5000
random_seed = 42

# =========================
# 1) Načítanie molekúl
# =========================
print(f"Načítavam molekuly zo súboru: {input_file}")
structures = read(input_file, index=":")
print(f"Načítaných molekúl: {len(structures)}")

if len(structures) < n_train + n_val:
    raise ValueError(
        f"Dataset je príliš malý. Potrebujem aspoň {n_train + n_val} molekúl, "
        f"ale našiel som len {len(structures)}."
    )

# =========================
# 2) Premiešanie molekúl
# =========================
random.seed(random_seed)
random.shuffle(structures)

# =========================
# 3) Výber train množiny
# =========================
train_structures = structures[:n_train]
remaining_structures = structures[n_train:]

print(f"Train množina: {len(train_structures)} molekúl")
print(f"Zvyšok po odobratí train množiny: {len(remaining_structures)} molekúl")

# =========================
# 4) Výber 5000 molekúl s najnižším DS zo zvyšku
# =========================
# Nižší DS = lepší (napr. viac negatívny)
remaining_sorted = sorted(
    remaining_structures,
    key=lambda mol: float(mol.get_potential_energy())
)

val_structures = remaining_sorted[:n_val]

print(f"Val množina: {len(val_structures)} molekúl")
print("Rozsah DS vo val množine:")
print(f"  min DS = {min(m.get_potential_energy() for m in val_structures):.6f}")
print(f"  max DS = {max(m.get_potential_energy() for m in val_structures):.6f}")

# =========================
# 5) Uloženie
# =========================
write(train_file, train_structures)
write(val_file, val_structures)

print(f"Train množina uložená do: {train_file}")
print(f"Val množina uložená do: {val_file}")