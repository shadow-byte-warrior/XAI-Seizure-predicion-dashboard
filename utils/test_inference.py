import mne
import numpy as np
import os
import torch
from src.data_loader import align_channels, normalize_channel
from src.ml_pipeline import prepare_sequences, run_inference, load_seizure_intervals, is_seizure, MODEL_CONFIGS

def test_inference(edf_path, seizures_path, model_name):
    print("=========================================")
    print(f"Testing Model: {model_name}")
    print(f"EDF File: {edf_path}")
    print("=========================================")
    
    # 1. Load EDF directly using MNE
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        normalized_names = {ch: normalize_channel(ch) for ch in raw.ch_names}
        raw.rename_channels(normalized_names)
        unique_channels = list(dict.fromkeys(raw.ch_names))
        raw.pick_channels(unique_channels)
        
        sfreq = raw.info['sfreq']
        duration = raw.n_times / sfreq
        print(f"File loaded. Duration: {duration:.2f}s, Sampling Rate: {sfreq}Hz")
    except Exception as e:
        print(f"Error loading EDF: {e}")
        return
    
    # 2. Align Channels
    config = MODEL_CONFIGS[model_name]
    aligned_data, _ = align_channels(raw, config['channels'])
    
    # 3. Prepare Sequences
    print("Extracting features and preparing sequences...")
    sequences, seq_times = prepare_sequences(aligned_data, sfreq, config)
    sequences = sequences[:10]
    seq_times = seq_times[:10]
    print(f"Prepared {len(sequences)} sequences.")
    
    # 4. Run Inference
    model_path = config['filename']
    print(f"Running inference using {model_path}...")
    predictions, probabilities, _, _, _ = run_inference(sequences, model_path, config, shap_enabled=True)
    
    # 5. Load Seizures and Validate
    try:
        with open(seizures_path, "rb") as f:
            content = f.read()
        intervals = load_seizure_intervals(content)
    except Exception as e:
        print(f"Error loading seizures file: {e}")
        intervals = None
    
    if intervals:
        correct = 0
        for t, pred in zip(seq_times, predictions):
            if pred == is_seizure(t, intervals):
                correct += 1
        accuracy = correct / len(predictions) * 100
        print(f"\nResults:")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Analyze seizure window
        gt_labels = np.array([is_seizure(t, intervals) for t in seq_times])
        sz_indices = np.where(gt_labels == 1)[0]
        if len(sz_indices) > 0:
            sz_probs = probabilities[sz_indices]
            print(f"Probabilities during seizure window (avg): {np.mean(sz_probs):.4f}")
            print(f"Max probability during seizure: {np.max(sz_probs):.4f}")
            print(f"Min probability during seizure: {np.min(sz_probs):.4f}")
            print(f"Detected {np.sum(predictions[sz_indices])} out of {len(sz_indices)} seizure segments.")
        print(f"Predicted Seizure Segments: {np.sum(predictions)}")
        print(f"Total Segments: {len(predictions)}")
        
        if np.sum(predictions) > 0:
            sz_times = seq_times[predictions == 1]
            print(f"Seizure predicted at times: {sz_times[:10]} ...")
    else:
        print("\nNo valid seizure intervals found for validation.")
        print(f"Predicted Seizure Segments: {np.sum(predictions)}")

if __name__ == "__main__":
    test_inference("chb02_16+.edf", "chb02_16_clean.seizures", "DA-GRL")
