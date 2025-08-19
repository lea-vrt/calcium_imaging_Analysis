#Calcium Imaging Analysis Pipeline (Healthy vs SMA)

This repository contains a full end-to-end pipeline for processing 2-photon calcium imaging recordings using Suite2p
, followed by signal preprocessing, feature extraction, and machine learning classification between Healthy and SMA conditions.
Although the current dataset is small (~8 recordings), the pipeline is modular and scalable, allowing for the easy addition of additional recordings.
📂 input_data/
┣ 📂 Healthy_recordings/
┗ 📂 SMA_recordings/

📂 suite2p_output/
┗ 📂 {RecordingName}/suite2p/plane0/
┣ 📄 F.npy
┣ 📄 Fneu.npy
┣ 📄 spks.npy
┣ 📄 iscell.npy
┣ 📄 stat.npy
┣ 📄 ops.npy
┣ 📄 F_corrected.npy
┣ 📄 F_dff.npy
┣ 📄 F_dff_smooth.npy
┣ 📄 spikes_filtered.csv
┗ 📄 calcium_features_suite2p_refined.csv

📂 models/
┣ 📄 Healthy_vs_SMA_best.joblib
┣ 📄 decision_threshold.txt
┗ 📄 recording_level_columns.json

📄 MASTER_cells_Healthy_vs_SMA.csv
📄 RECORDING_LEVEL.csv
📄 run_suite2p.py
📄 preprocess_signals.py
📄 extract_features.py
📄 analyze_correlations.py
📄 train_classifier.py
📄 README.md
