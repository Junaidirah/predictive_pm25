# Final Project – Environment Setup Documentation

This document describes setup required for the **PM2.5 research project** using Python and Conda.

---

## 🔧 Preparation & Setup

### Create Conda Environment
Create a new Conda environment with Python 3.10:

```bash
conda create -n myvnev python=3.10
```
u can change the name myvnev
###  Create Conda Environment
```bash
conda activate myvnev
```
### Install library and model
The following libraries are used for data manipulation, visualization, and preprocessing:

- NumPy & Pandas 
- Matplotlib & Seaborn 
- Scikit-learn
- tensorflow
- lightgbm
- xgboost
- optuna

All dependencies are managed using a requirements.txt file.
```bash
pip install -r requirements.txt
```

### Deactivating the Environment
```
conda deactivate myvnev
```
---
