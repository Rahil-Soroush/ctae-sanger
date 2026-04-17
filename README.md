# CTAE: Coupled Transformer Autoencoder for Disentangling Multiregion Neural Latent Dynamics

## Overview

This repository contains code for training and evaluating the Coupled Transformer Autoencoder (CTAE), a model designed to learn shared and region-specific latent representations from multi-region neural recordings.

The model is demonstrated on simultaneous electrophysiology recordings from motor cortex regions (M1 and PMd), and supports:

- learning shared vs region-specific dynamics
- latent space analysis and visualization
- decoding behavioral variables such as direction and position

---

## Data

This project uses the publicly available multi-region electrophysiology dataset from the DANDI Archive.

**Dandiset:** 000688  
https://dandiarchive.org/dandiset/000688/0.250122.1735/files?location=sub-M&page=1

Please download the following file:

- `sub-M_ses-CO-20140303_behavior+ecephys.nwb`

After downloading, place the file inside the project directory:

- `./data/raw`

---

## Repository Structure

data/
  raw/                # Raw NWB files
models/
  __init__.py
  ctae.py             # CTAE model definition
trained_models/       # Saved trained models
data_preprocessing.ipynb
train.py              # Training script
evaluate_ctae_latents.ipynb
utils.py              # Helper functions
README.md

## Repository Structure

data/
  raw/                # Raw NWB files
models/
  __init__.py
  ctae.py             # CTAE model definition
trained_models/       # Saved trained models
data_preprocessing.ipynb
train.py              # Training script
evaluate_ctae_latents.ipynb
utils.py              # Helper functions
README.md

---

## Usage

### 1. Data Preprocessing

Run:
data_preprocessing.ipynb

This extracts:
- trial-aligned firing rates
- M1 and PMd neural activity
- behavioral variables such as direction and position

Processed data is saved in:
./data/processed/

---

### 2. Training the Model

Note: This step can be skipped if you only want to reproduce the results from the paper. Pretrained models are already provided in `./trained_models/`, and you can directly proceed to the evaluation step.

Run:
python train.py

This will:
- train the CTAE model on M1-PMd data
- optimize reconstruction, alignment, and orthogonality losses
- save the best model based on validation performance

Trained models are saved in:
./trained_models/

---

### 3. Evaluation and Analysis

Run:
evaluate_ctae_latents.ipynb

This notebook includes:
- visualization of latent trajectories
- decoding of target direction
- decoding of paw position

---

## Model Summary

CTAE learns a structured latent space composed of:
- shared subspace capturing dynamics common across regions
- region-specific subspaces capturing region-unique activity

The training objective includes:
- reconstruction loss
- reconstruction from shared latents alone
- shared latent alignment
- orthogonality constraint between and within subspaces

---


## Citation

If you use this code, please cite:
[https://openreview.net/forum?id=oeoCgcYIyf]

---

