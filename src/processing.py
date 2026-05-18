import numpy as np
from scipy import signal

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def preprocess_signal(data, sfreq, lowcut=0.5, highcut=45.0, apply_notch=True):
    """
    Preprocesses the raw EEG signal by applying bandpass and optional notch filtering.
    """
    # Bandpass filter
    filtered_data = butter_bandpass_filter(data, lowcut, highcut, sfreq, order=4)
    
    # Optional Notch filter (e.g., 50Hz or 60Hz power line noise)
    if apply_notch:
        # We use a notch filter at 50Hz as an example (common in EU/Asia, 60Hz in US)
        # However, since our highcut is 45Hz, it shouldn't be strictly necessary.
        # But we'll include it for completeness if the user increases highcut.
        nyq = 0.5 * sfreq
        freq = 50.0 / nyq
        b, a = signal.iirnotch(freq, 30.0)
        filtered_data = signal.filtfilt(b, a, filtered_data)
        
    return filtered_data

def normalize_signal(data):
    """
    Normalizes the signal to have zero mean and unit variance (Z-score normalization).
    """
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data
    return (data - mean) / std

def extract_bands(data, sfreq):
    """
    Extracts standard EEG frequency bands from the given signal.
    """
    bands = {
        'Delta (0.5-4 Hz)': (0.5, 4.0),
        'Theta (4-8 Hz)': (4.0, 8.0),
        'Alpha (8-13 Hz)': (8.0, 13.0),
        'Beta (13-30 Hz)': (13.0, 30.0),
        'Gamma (30-45 Hz)': (30.0, 45.0)
    }
    
    band_signals = {}
    for band_name, (low, high) in bands.items():
        band_signals[band_name] = butter_bandpass_filter(data, low, high, sfreq, order=3)
        
    return band_signals

def ensure_montage(raw):
    """
    Ensures the raw object has electrode positions (dig) for topographic mapping.
    Applies standard 10-20 montage with fuzzy name matching and cleaning.
    Specifically handles bipolar pairs like 'FP1-F7' by taking the first electrode.
    """
    import mne
    try:
        # 1. Clean and Normalize Channel Names
        def clean_name(ch):
            # Split bipolar pairs (e.g. FP1-F7 -> FP1)
            name = str(ch).upper().split('-')[0]
            # Strip common EEG prefixes/suffixes
            name = name.replace("EEG", "").replace("REF", "").replace("LE", "").strip()
            # Standard mappings
            mapping = {
                'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
                'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8' # 10-20 to 10-10 conversion
            }
            return mapping.get(name, name)

        # Handle duplicates during renaming
        new_names = []
        seen = set()
        for ch in raw.ch_names:
            base = clean_name(ch)
            final = base
            counter = 2
            while final in seen:
                final = f"{base}_{counter}"
                counter += 1
            new_names.append(final)
            seen.add(final)
            
        raw.rename_channels(dict(zip(raw.ch_names, new_names)))
        
        # 2. Apply Standard Montage
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing='ignore', match_case=False)
        
        # 3. Validation Layer
        if raw.info.get('dig') is None or len(raw.info.get('dig', [])) == 0:
            return raw, "⚠️ Electrode positions missing after montage application."
            
        return raw, None
    except Exception as e:
        return raw, f"❌ Montage Error: {str(e)}"
