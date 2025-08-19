import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


BASE_OUTPUT = os.path.abspath('suite2p_output')

def discover_plane0_folders(base_output):
    ordered = []
    for cls in ['Healthy', 'SMA']:  # Healthy first, then SMA
        prefix = f"{cls}_"
        for d in sorted(os.listdir(base_output)):
            if not d.startswith(prefix):
                continue
            plane0 = os.path.join(base_output, d, 'suite2p', 'plane0')
            if os.path.isdir(plane0):
                ordered.append((cls, d, plane0))
    return ordered

for cls, rec_name, plane0_folder in discover_plane0_folders(BASE_OUTPUT):

    print(f"\nProcessing denoising for: {plane0_folder}  (class={cls}, recording={rec_name})")

    try:
        # Load Suite2p output files
        F_raw = np.load(os.path.join(plane0_folder, 'F.npy'))  # Fluorescence signals
        Fneu = np.load(os.path.join(plane0_folder, 'Fneu.npy'))  # Background neuropil
        spks = np.load(os.path.join(plane0_folder, 'spks.npy'))  # Inferred spikes
        iscell = np.load(os.path.join(plane0_folder, 'iscell.npy'))  # REAL AND NOT REAL NEURONS
        stat = np.load(os.path.join(plane0_folder, 'stat.npy'), allow_pickle=True)  # LOCATION OFCOURSE COORDINATES
        ops = np.load(os.path.join(plane0_folder, 'ops.npy'), allow_pickle=True).item()
        mean_img = ops['meanImg']  # MEAN OF MOTION CORRECTION

        # Plot mean registered image
        # plt.figure(figsize=(6, 6))
        # plt.imshow(mean_img, cmap='gray')
        # plt.title('Mean Registered Image')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

        # Print accepted ROI indices
        accepted = np.where(iscell[:, 0] == 1)[0]
        print("Accepted cell indices:", accepted)

        # Keep only accepted cells
        F_raw = F_raw[accepted]
        Fneu = Fneu[accepted]
        spks = spks[accepted]
        stat = stat[accepted]

        # Spatial masks of first 5 ROIs
        # for i in range(min(5, len(stat))):
        #     mask = np.zeros_like(mean_img)
        #     s = stat[i]
        #     mask[s['ypix'], s['xpix']] = s['lam']
        #     plt.figure()
        #     plt.imshow(mask, cmap='hot')
        #     plt.colorbar()
        #     plt.title(f'ROI {i} Spatial Mask')
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.show()

        # Neuropil correction
        neuropil_factor = 0.7
        F_corrected = F_raw - neuropil_factor * Fneu

        # Compute baseline (F0) and ΔF/F
        F0 = np.percentile(F_corrected, 8, axis=1, keepdims=True)  # BASELINE VALUE FOR EACH NEURON
        F_dff = (F_corrected - F0) / F0  # NORMALIZE SIGNAL CHANGE SHAPE, COLUMN , ROW TIMEPOINT

        # Smooth ΔF/F traces
        F_dff_smooth = gaussian_filter1d(F_dff, sigma=1, axis=1)

        # Plot raw vs neuropil-corrected for one cell
        # plt.figure(figsize=(10, 4))
        # plt.plot(F_raw[0], label='Raw F', alpha=0.6)
        # plt.plot(F_corrected[0], label='Corrected F', linewidth=2)
        # plt.xlabel('Frame')
        # plt.ylabel('Fluorescence')
        # plt.title('Raw vs Neuropil-Corrected Fluorescence (Cell 0)')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        # Plot first 10 ΔF/F traces
        # plt.figure(figsize=(16, 6))
        # for i in range(min(10, len(F_dff_smooth))):
        #     plt.plot(F_dff_smooth[i], label=f'Cell {i}')
        # plt.xlabel('Frame')
        # plt.ylabel('ΔF/F')
        # plt.title('Extracted Calcium Traces (Smoothed dF/F)')
        # plt.legend(ncol=2, fontsize=8)
        # plt.tight_layout()
        # plt.show()

        # Plot ΔF/F and inferred spikes for selected cells
        # cell_indices_to_plot = [0, 1, 2]
        # for cell_idx in cell_indices_to_plot:
        #     plt.figure(figsize=(12, 4))
        #     plt.plot(F_dff[cell_idx], label='dF/F')
        #     plt.plot(spks[cell_idx] * np.max(F_dff[cell_idx]), label='Inferred Spikes', alpha=0.7)
        #     plt.xlabel('Frame')
        #     plt.ylabel('Signal')
        #     plt.title(f'Calcium Trace vs Spike Train (Cell {cell_idx})')
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.show()

        # Save processed arrays
        np.save(os.path.join(plane0_folder, 'F_corrected.npy'), F_corrected)
        np.save(os.path.join(plane0_folder, 'F_dff.npy'), F_dff)
        np.save(os.path.join(plane0_folder, 'F_dff_smooth.npy'), F_dff_smooth)
        np.save(os.path.join(plane0_folder, 'spks_filtered.npy'), spks)

        # Save as CSV for easy viewing
        pd.DataFrame(F_dff_smooth.T).to_csv(os.path.join(plane0_folder, 'F_dff_smooth.csv'), index=False)
        pd.DataFrame(spks.T).to_csv(os.path.join(plane0_folder, 'spikes_filtered.csv'), index=False)

        print("Denoising complete. Saved ΔF/F and spike data.")

    except Exception as e:
        print(f" Error processing {plane0_folder}: {e}")
