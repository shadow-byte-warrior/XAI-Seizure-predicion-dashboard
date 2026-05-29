import mne
import numpy as np
import tempfile
import os

def normalize_channel(ch):
    """
    Normalizes channel names to a standard format (uppercase, no spaces).
    """
    return ch.strip().upper().replace(" ", "")

def load_edf(uploaded_file):
    """
    Loads an EDF file using mne-python from a Streamlit UploadedFile object.
    
    Args:
        uploaded_file: Streamlit UploadedFile object containing the EDF file.
        
    Returns:
        dict: A dictionary containing the raw mne object, channel names, 
              sampling frequency, duration, and the raw data array.
    """
    # Create a unique temporary file to avoid concurrent session conflicts
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name
        
    try:
        raw = mne.io.read_raw_edf(temp_file_path, preload=True, verbose=False)
        
        # 1. Normalize Channel Names while strictly guaranteeing uniqueness
        normalized_names = {}
        seen = set()
        for ch in raw.ch_names:
            norm = normalize_channel(ch)
            final_norm = norm
            counter = 1
            while final_norm in seen:
                final_norm = f"{norm}_{counter}"
                counter += 1
            normalized_names[ch] = final_norm
            seen.add(final_norm)
            
        raw.rename_channels(normalized_names)
        
        # 2. Remove Duplicates
        unique_channels = list(dict.fromkeys(raw.ch_names))
        raw.pick_channels(unique_channels)
        
        # Extract basic info
        ch_names = raw.ch_names
        sfreq = raw.info['sfreq']
        n_samples = raw.n_times
        duration = n_samples / sfreq
        
        data, times = raw[:, :]
        
        return {
            'raw': raw,
            'ch_names': ch_names,
            'sfreq': sfreq,
            'duration': duration,
            'data': data,
            'times': times,
            'n_samples': n_samples
        }
    except Exception as e:
        raise ValueError(f"Failed to load EDF file: {str(e)}")
    finally:
        # Securely delete the temporary file from the disk
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


STANDARD_CHANNELS = [
    "FP1-F7","F7-T7","T7-P7","P7-O1",
    "FP1-F3","F3-C3","C3-P3","P3-O1",
    "FP2-F4","F4-C4","C4-P4","P4-O2",
    "FP2-F8","F8-T8","T8-P8","P8-O2",
    "FZ-CZ","CZ-PZ"
]

def align_channels(raw, required_channels_count=18):
    """
    Aligns the raw data to standard channels, filling missing ones with zeros.
    """
    data = raw.get_data()
    ch_names = [ch.upper().strip() for ch in raw.ch_names]
    
    # If model needs 18, use strictly the 18 STANDARD_CHANNELS.
    # If model needs 23, use the 18 + first 5 other available channels to reach 23.
    target_channels = STANDARD_CHANNELS.copy()
    
    if required_channels_count > len(target_channels):
        extra_needed = required_channels_count - len(target_channels)
        extra_channels = [ch for ch in ch_names if ch not in target_channels]
        target_channels.extend(extra_channels[:extra_needed])
        
    # Pad if we still don't have enough target names to search for
    while len(target_channels) < required_channels_count:
        target_channels.append(f"DUMMY_{len(target_channels)}")

    aligned_data = []

    for ch in target_channels[:required_channels_count]:
        if ch in ch_names:
            idx = ch_names.index(ch)
            aligned_data.append(data[idx])
        else:
            print(f"[WARNING] Missing channel: {ch} -> filling with zeros")
            aligned_data.append(np.zeros(data.shape[1]))

    return np.array(aligned_data), target_channels[:required_channels_count]

