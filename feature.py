import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import zscore
import os

BASE_OUTPUT = os.path.abspath('suite2p_output')
def discover_plane0_folders(base_output):
    ordered = []
    for cls in ['Healthy', 'SMA']: 
        prefix = f"{cls}_"
        for d in sorted(os.listdir(base_output)):
            if not d.startswith(prefix):
                continue
            plane0 = os.path.join(base_output, d, 'suite2p', 'plane0')
            if os.path.isdir(plane0):
                ordered.append((cls, d, plane0))
    return ordered


fs = 10.0  
min_spikes_per_burst = 3
burst_interval_threshold = 1.0 
spike_min_distance = int(1.5 * fs)  
zscore_peak_height = 4           
zscore_peak_prominence = 1.5       

for cls, rec_name, suite2p_path in discover_plane0_folders(BASE_OUTPUT):
    try:
        print(f"\n=== Feature extraction for: {cls}/{rec_name} ===")
    
        spks = np.load(os.path.join(suite2p_path, 'spks.npy'))        
        iscell = np.load(os.path.join(suite2p_path, 'iscell.npy'))     

        accepted = np.where(iscell[:, 0] == 1)[0]
        spks = spks[accepted]  

   
        F_dff = pd.read_csv(os.path.join(suite2p_path, 'F_dff_smooth.csv')).to_numpy().T
        assert spks.shape[0] == F_dff.shape[0], "Mismatch between spike and dF/F data"

        timestamps = np.arange(spks.shape[1]) / fs

        features = []

        for i in range(spks.shape[0]):
            spikes = spks[i]
            dff = F_dff[i]

            z_spikes = zscore(spikes)
            peak_indices, _ = find_peaks(
                z_spikes,
                height=zscore_peak_height,
                distance=spike_min_distance,
                prominence=zscore_peak_prominence
            )
            spike_times = timestamps[peak_indices]

            # ISI and burst detection
            isis = np.diff(spike_times)
            bursts = []
            current = []

            for t in spike_times:
                if not current:
                    current.append(t)
                elif t - current[-1] <= burst_interval_threshold:
                    current.append(t)
                else:
                    if len(current) >= min_spikes_per_burst:
                        bursts.append(current)
                    current = [t]
            if len(current) >= min_spikes_per_burst:
                bursts.append(current)

            burst_durations = [b[-1] - b[0] for b in bursts]
            spikes_per_burst = [len(b) for b in bursts]

            features.append({
                'cell_index': i,
                'roi_index': int(accepted[i]),
                'spike_count': len(spike_times),
                'firing_rate_hz': len(spike_times) / (len(spikes) / fs),
                'mean_dff': float(np.mean(dff)),
                'max_dff': float(np.max(dff)),
                'mean_isi': float(np.mean(isis)) if len(isis) > 0 else np.nan,
                'cv_isi': float(np.std(isis) / np.mean(isis)) if len(isis) > 1 and np.mean(isis) > 0 else np.nan,
                'burst_count': len(bursts),
                'mean_burst_duration': float(np.mean(burst_durations)) if burst_durations else np.nan,
                'mean_spikes_per_burst': float(np.mean(spikes_per_burst)) if spikes_per_burst else np.nan
            })

    
        df_features = pd.DataFrame(features)
        out_csv = os.path.join(suite2p_path, 'calcium_features_suite2p_refined.csv')
        df_features.to_csv(out_csv, index=False)
        print(f"Feature extraction complete. Output saved to {out_csv}")

    except Exception as e:
        print(f" Error processing {suite2p_path}: {e}")
