# Calcium Imaging Analysis Pipeline (Healthy vs SMA)

The aim of this project is to propose a tool capable of performing automated feature extraction and machine-learning based classification of motor neuron calcium imaging data to accurately distinguish between healthy and pathological behavior. 
This repository provides a **complete end-to-end pipeline** for analyzing 2-photon calcium imaging data, with features based on [Suite2p](https://github.com/MouseLand/suite2p).  
It includes:

- ROI extraction & spike inference (Suite2p)
- ROI correlation network 
- Neuropil correction & ΔF/F computation  
- Spike/burst detection and feature extraction  
- Recording-level feature aggregation  
- Machine learning classification (Healthy vs SMA)  

The pipeline is still currently underdevelopment and aims to be modular and scalable → adding more recordings is straightforward.
It is important to note that so far, the database is only built from ~8 video recordings and will be enhanced in near future for better accuracy and less overfitting.

---

##  Project Structure

```
input_data/                
 ├── Healthy_recordings/        # Raw .tif recordings (Healthy group)
 └── SMA_recordings/            # Raw .tif recordings (SMA group)

suite2p_output/                 # Outputs per recording
 └── {RecordingName}/suite2p/plane0/
     ├── F.npy                  # Raw fluorescence
     ├── Fneu.npy               # Neuropil background
     ├── spks.npy               # Inferred spikes
     ├── iscell.npy             # ROI classification (cell vs not-cell)
     ├── stat.npy               # ROI spatial masks
     ├── ops.npy                # Suite2p metadata
     ├── F_corrected.npy        # Neuropil-corrected signals
     ├── F_dff.npy              # ΔF/F traces
     ├── F_dff_smooth.npy       # Smoothed ΔF/F
     ├── spikes_filtered.csv
     └── calcium_features_suite2p_refined.csv

models/                         # Trained ML models
 ├── Healthy_vs_SMA_best.joblib
 ├── decision_threshold.txt
 └── recording_level_columns.json

MASTER_cells_Healthy_vs_SMA.csv # All cell-level features
RECORDING_LEVEL.csv             # Aggregated per-recording features

run_suite2p.py                  # Runs Suite2p on raw data
preprocess_signals.py           # Neuropil correction + ΔF/F
extract_features.py             # Spike detection + feature extraction
analyze_correlations.py         # ROI correlation networks
train_classifier.py             # ML model training & evaluation
README.md
```

---

## Installation
**(for Windows users) All installation steps must be run from the computer's command prompt**

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sania-2000/calcium_imaging_Analysis.git
   cd calcium-pipeline
   ```

2. **Create environment (conda recommended)**
   ```bash
   conda create -n calpipe python=3.10
   conda activate calpipe
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Prepare Input Data
Place `.tif` recordings into:
```
input_data/Healthy_recordings/
input_data/SMA_recordings/
```

### 2. Run Suite2p
```bash
python run_suite2p.py
```

### 3. Preprocess Signals
```bash
python preprocess_signals.py
```
- Neuropil correction  
- ΔF/F computation  
- Saves `.npy` + `.csv` files  

### 4. Extract Features
```bash
python extract_features.py
```
- Spike counts, firing rate  
- ΔF/F statistics (mean, max)  
- ISI & burst statistics  

### 5. Train & Evaluate Classifier
```bash
python train_classifier.py
```
- Aggregates features → recording level  
- Splits train/test (stratified by recording)  
- Trains RF / Logistic Regression  
- Saves best model + threshold  

---

## Outputs

- **Cell-level features** → `calcium_features_suite2p_refined.csv`  
- **Recording-level features** → `RECORDING_LEVEL.csv`  
- **Model & threshold** → in `models/`  
- **Predictions** → per-recording true vs predicted labels  

---

##  Workflow Overview

```
 Raw .tif recordings
        ↓
      Suite2p
        ↓
 Preprocessing (ΔF/F, spikes)
        ↓
 Feature Extraction (cell-level)
        ↓
 Recording-level aggregation
        ↓
 ML Classification (Healthy vs SMA)
```

---

##  Notes

- So far, ML results are highly unreliable → use mainly for **pipeline demonstration**.  
- Designed to be **scalable**: just drop new recordings into `input_data/` and re-run.  
- Features and models can be extended for larger datasets.  

---

## Next Steps

- Add more recordings to improve classifier performance  
- Explore additional network/synchrony features  
- Compare classical ML with deep learning approaches  

 
