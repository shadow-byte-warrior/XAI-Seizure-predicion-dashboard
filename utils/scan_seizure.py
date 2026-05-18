import mne
import numpy as np
import torch
import os
import pandas as pd
from src.data_loader import align_channels, normalize_channel
from src.ml_pipeline import prepare_sequences, run_inference, MODEL_CONFIGS

def scan_file(edf_path, model_name="DA-GRL", target_start=2996, target_end=3036):
    print(f"\n--- Scanning: {edf_path} ---")
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        normalized_names = {ch: normalize_channel(ch) for ch in raw.ch_names}
        raw.rename_channels(normalized_names)
        unique_channels = list(dict.fromkeys(raw.ch_names))
        raw.pick_channels(unique_channels)
        
        config = MODEL_CONFIGS[model_name]
        aligned_data, _ = align_channels(raw, config['channels'])
        
        sfreq = raw.info['sfreq']
        sequences, seq_times = prepare_sequences(aligned_data, sfreq, config, window_size=1024, step_sec=2.0)
        
        model_path = config['filename']
        batch_size = 100
        all_preds = []
        all_probs = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            preds, probs, _, _, _ = run_inference(batch, model_path, config, shap_enabled=False)
            all_preds.extend(preds)
            all_probs.extend(probs)
            
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # Calculate Smoothed Risk Score (same as in app.py)
        smoothed_probs = pd.Series(all_probs).rolling(window=5, center=True).mean().fillna(0).values
        risk_score = np.max(smoothed_probs)
        
        print(f"Total sequences: {len(all_preds)}")
        print(f"Raw Detections (conf > 0.5): {np.sum(all_preds)}")
        print(f"Max Raw Probability: {np.max(all_probs):.4f}")
        print(f"Max Smoothed Probability (Risk Score): {risk_score:.4f}")
        
        # Check seizure window
        sz_indices = np.where((seq_times >= target_start) & (seq_times <= target_end))[0]
        if len(sz_indices) > 0:
            sz_probs = all_probs[sz_indices]
            sz_smoothed = smoothed_probs[sz_indices]
            print(f"Seizure Window Raw Avg Prob: {np.mean(sz_probs):.4f}")
            print(f"Seizure Window Smoothed Max: {np.max(sz_smoothed):.4f}")
        
    except Exception as e:
        print(f"Error scanning {edf_path}: {e}")

if __name__ == "__main__":
    seizure_file = r"seizure\chb01_03.edf"
    scan_file(seizure_file)
