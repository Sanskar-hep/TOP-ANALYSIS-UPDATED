<div align="center">

# TOP Analysis — Electron + Jets Channel

**Coffea-based framework for charge asymmetry extraction in tt̄ production**

![Python](https://img.shields.io/badge/Python-3.8-blue?style=flat-square&logo=python&logoColor=white)
![Coffea](https://img.shields.io/badge/Coffea-latest-teal?style=flat-square)
![CMS](https://img.shields.io/badge/CMS-Run%202%20UL-blue?style=flat-square)
![NanoAOD](https://img.shields.io/badge/Format-NanoAOD-green?style=flat-square)

</div>

---

## Overview

This repository contains a **Coffea-based analysis framework** for the **Electron + Jets channel** in top quark pair production. The framework extracts **flavour-separated, collider-independent charge asymmetries** (Aᵤ and A_d) in tt̄ production using CMS Run 2 Ultra-Legacy data.

---

## ✨ What This Framework Does

| Module | Description |
|--------|-------------|
| 🔍 **Event & Object Selection** | Selection Cuts on electrons, Jets and MET|
| 📊 **Data / MC Comparison** | Data-MC agreement studies for each era of Run2 |
| 🧮 **ABCD QCD Estimation** | Data-driven QCD background estimation using the ABCD method |
| 🤖 **BDT Classification** | XGBoost-based binary classifier for distinguishing top quark pair arising from qq̄(signal) from those of gg|
| 📐 **χ² Kinematic Fitting** | Reconstruction of the top-quark and the antitop quark kinematic observables|
| 📈 **TUnfold Unfolding** | Response matrix construction and unfolding pipeline for parton-level asymmetries (Will be updated in a few weeks) |

---

## 🔬 Analysis Pipeline

```
NanoAOD Input
     │
     ▼
Event & Object Selection (SFs, triggers, weights)
     │
     ▼
ABCD QCD Background Estimation
     │
     ▼
BDT Training (qq̄ / gg enrichment)
     │
     ▼
χ² Kinematic Fitting (top quark reconstruction)
     │
     ▼
TUnfold Unfolding (response matrix, L-curve scan)
     │
     ▼
Aᵤ and A_d — Parton-level Charge Asymmetries
```

---

## 📦 Dependencies

The analysis requires the following Python packages:

| Package | Purpose |
|---------|---------|
| [coffea](https://coffea-hep.readthedocs.io/en/latest/) | columnar HEP analysis framework |
| [awkward-array](https://pypi.org/project/awkward/) | jagged array operations |
| [dask-awkward](https://docs.dask.org/en/latest/) | lazy/distributed execution |
| [hist](https://hist.readthedocs.io/en/latest/) | histogram objects |
| [correctionlib](https://cms-nanoaod.github.io/correctionlib/install.html) | scale factor corrections (b-tagging, PU, etc.) |
| [numpy](https://numpy.org/) | numerical operations |
| [pyarrow](https://pypi.org/project/pyarrow/) | parquet I/O for BDT inputs |
| [xgboost](https://xgboost.readthedocs.io/) | gradient boosted decision trees |
| [uproot](https://uproot.readthedocs.io/) | ROOT I/O in Python |

---

## 🚀 Installation

It is recommended to use a virtual environment (e.g., Conda).

### Using Conda

```bash
# Create the environment
conda create -n <env_name> python=3.8 \
  conda-forge::uproot \
  conda-forge::root \
  conda-forge::awkward \
  conda-forge::coffea \
  conda-forge::pyarrow

# Activate
conda activate <env_name>

# Install remaining dependencies
pip install correctionlib hist xgboost
```

---


## 📋 Dataset

- **Format**: NanoAOD (CMS Run 2 Ultra-Legacy)
- **Channel**: Electron + Jets
- **Eras**: 2016 (preVFP + postVFP), 2017, 2018
- **Collaboration**: CMS

---

<div align="center">

**CMS Collaboration · Run 2 UL · Electron + Jets · tt̄ Charge Asymmetry**


