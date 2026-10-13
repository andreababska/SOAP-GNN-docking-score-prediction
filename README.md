# 🧬 Development and Optimization of Graph Neural Network Models for Molecular Docking Score Prediction

Master's thesis code for docking score prediction using SOAP, DimeNet++, and GemNet on SARS-CoV-2 main protease ligands.

Molecular docking is accurate but slow, which becomes a bottleneck when screening libraries of billions of molecules. This project trains machine learning models to predict docking scores directly from molecular structure, replacing the expensive docking step with fast inference.

## 🛠️ Technologies

- **Language:** Python
- **Deep learning:** PyTorch, PyTorch Geometric
- **Models:** SOAP descriptors + feed-forward neural network (FFNN), DimeNet++, GemNet
- **Hyperparameter optimization:** Hyperopt (TPE)
- **Docking reference:** AutoDock Vina scores
- **Target:** SARS-CoV-2 main protease (3CLpro, PDB ID: 6WQF)
- **Compute:** Devana HPC cluster (Computing Centre of the Slovak Academy of Sciences)

## ✨ Features

- Three benchmarked pipelines for docking score prediction: SOAP + FFNN, DimeNet++, and GemNet
- Automated hyperparameter search with Hyperopt TPE
- Early stopping and per-epoch CSV logging (`epoch;train_mse;val_mse;best_val_mse;train_time_s;val_time_s`)
- Two validation setups: Variant A (standard validation) and Variant B (validation targeted at low, i.e. best, docking scores)
- Scripts for visualizing docking score distributions

## 🔬 The Process

1. **Data:** Molecular structures in XYZ format (`in_vivo.xyz`, `in_vitro.xyz`, from Bucinsky et al. / Dryad). The `in_vivo` set contains 59,884 molecules with docking scores from -14.84 to -1.03 kcal/mol.
2. **Model design:** Compared a descriptor-based approach (SOAP + FFNN) with two geometric graph neural networks (DimeNet++ and GemNet).
3. **Training and optimization:** Models were trained in PyTorch / PyTorch Geometric, with Hyperopt used to tune hyperparameters. Large-scale runs were executed on the Devana supercomputer, including handling of CUDA out-of-memory errors.
4. **Evaluation:** Models were compared on global metrics (R², MSE) and on how well they rank the best-scoring molecules, which matters most for virtual screening.

## 📊 Results

- Global accuracy is essentially the same for all three models, but computational cost differs by orders of magnitude (minutes for SOAP vs. many hours for GemNet per fold).
- DimeNet++ with Variant B is the only configuration that clearly improves the strongly binding region and lowers fold-to-fold variability.
- All models underpredict the strongest binders, and they agree with each other more than with the reference, suggesting the residual error comes mainly from the data.
- Recommended for virtual screening: DimeNet++ with Variant B.

## 💡 What I Learned

- Building and tuning graph neural networks on 3D molecular data
- Running and debugging large-scale training on an HPC cluster (memory limits, early stopping, reproducible logging)
- Why global metrics alone are not enough: a model can score well overall and still be weaker where it matters (the best-scoring molecules)
- Structuring a long-running research project from data preparation to evaluation and written thesis

## 👩‍💻 Author

Andrea Bábska, STU FCHPT Bratislava
Supervisor: Ing. Marián Gall, PhD
