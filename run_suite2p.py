
import os
import glob
from suite2p import run_s2p
BASE_INPUT = os.path.abspath('input_data')          
BASE_OUTPUT = os.path.abspath('suite2p_output')    

FS = 10.0  
OPS_TEMPLATE = {
    'fs': FS,
    'nplanes': 1,
    'nchannels': 1,
    'functional_chan': 1,
    'tau': 1.0,
    'batch_size': 200,
    'nbinned': 500,
    'sparse_mode': True,
    'preclassify': 0,
    'allow_overlap': False,
    'threshold_scaling': 0.2,
    'spatial_scale': 1.0,
    'max_overlap': 0.5,
    'anatomical_only': 0,
    'do_registration': 1,
    'roidetect': True,
    'neuropil_extract': True,
    'spike_deconvolution': True,
    'diameter': [5, 6],
    'cellprob_threshold': 0.6,
    'use_builtin_classifier': False,
}

def process_folder(class_name: str, subdir: str):
    """
    class_name: 'Healthy' or 'SMA' (used in output naming)
    subdir: folder name inside input_data (e.g., 'Healthy_recordings')
    """
    input_folder = os.path.join(BASE_INPUT, subdir)
    tif_files = sorted(glob.glob(os.path.join(input_folder, '*.tif')))
    print(f"\n=== {class_name}: found {len(tif_files)} tif(s) in {input_folder} ===")

    if not tif_files:
        return

    for tif_path in tif_files:
        tif_name = os.path.splitext(os.path.basename(tif_path))[0]
        output_folder = os.path.join(BASE_OUTPUT, f"{class_name}_{tif_name}")
        os.makedirs(output_folder, exist_ok=True)

        ops = dict(OPS_TEMPLATE) 
        ops.update({
            'data_path': [os.path.dirname(tif_path)],
            'save_path0': output_folder,
            'tiff_list': [tif_path],
        })

        print(f"Processing {class_name}/{tif_name} ...")
        run_s2p(ops=ops)
        print(f"Finished {class_name}/{tif_name} → {output_folder}")
if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT, exist_ok=True)

    process_folder(class_name='Healthy', subdir='Healthy_recordings')

    process_folder(class_name='SMA', subdir='SMA_recordings')

   
