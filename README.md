# SynthDoppler-Generation

## Auto_FEDUS

This repository contains code and resources for the **Auto-FEDUS** model to generated heartbeat DUS signals from input FECGs.

## Repository Structure
**`data/`** contains notebooks for data wrangling and preperation.
Folders ending with _ecg are the ones we are actively working on to reporting the results in the paper.
**`Auto-FEDUS/`** is the proposed model. We developed other generative approaches for comparison.

Each folder contains the following components:

- **`logs/`**: Contains saved training logs.
- **`models/`**: Contains saved model checkpoints.
- **`plots/`**: Stores output visualizations.
- **`src/`**: Includes the main scripts for model development.

## Requirements

To run this project, ensure you have the following installed:

- Python 
- Jupyter Notebook
- TensorFlow
- pywt

Link to the paper:
