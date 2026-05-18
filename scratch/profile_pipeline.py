import time
import mne
import numpy as np
import torch
from src.data_loader import align_channels, normalize_channel
from src.ml_pipeline import prepare_sequences, run_inference, MODEL_CONFIGS

def profile():
    edf_path = "chb02_16+.edf"
    model_name = "DA-GRL"
    
    print("Loading EDF...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    normalized_names = {ch: ch.strip().upper().replace(" ", "") for ch in raw.ch_names}
    raw.rename_channels(normalized_names)
    
    config = MODEL_CONFIGS[model_name]
    aligned_data, _ = align_channels(raw, config['channels'])
    sfreq = raw.info['sfreq']
    
    print(f"Profiling prepare_sequences for {len(aligned_data[0])} samples...")
    start = time.time()
    sequences, seq_times = prepare_sequences(aligned_data, sfreq, config)
    end = time.time()
    print(f"prepare_sequences took: {end - start:.2f}s")
    
    print(f"Profiling run_inference (SHAP=True) for {len(sequences)} sequences...")
    model_path = config['filename']
    start = time.time()
    predictions, probabilities, _, _, _ = run_inference(sequences, model_path, config, shap_enabled=True)
    end = time.time()
    print(f"run_inference took: {end - start:.2f}s")

if __name__ == "__main__":
    profile()
