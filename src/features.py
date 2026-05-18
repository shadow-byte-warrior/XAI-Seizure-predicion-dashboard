import numpy as np
from scipy import signal, stats
import pandas as pd

def extract_features(data, sfreq):
    """
    Extracts time-domain statistical features and basic frequency domain features.
    """
    # Shannon Entropy helper
    def shannon_entropy(x):
        hist, bin_edges = np.histogram(x, bins=20, density=True)
        return stats.entropy(hist + 1e-9)

    features = {
        'Mean': np.mean(data),
        'Variance': np.var(data),
        'Std Dev': np.std(data),
        'Min': np.min(data),
        'Max': np.max(data),
        'Skewness': stats.skew(data),
        'Kurtosis': stats.kurtosis(data),
        'Entropy': shannon_entropy(data),
        'RMS': np.sqrt(np.mean(data**2))
    }
    return features

def compute_psd(data, sfreq):
    """
    Computes Power Spectral Density using Welch's method.
    """
    # Use a window length of 4 seconds if possible, else the whole data length
    nperseg = min(int(4 * sfreq), len(data))
    
    freqs, psd = signal.welch(data, sfreq, nperseg=nperseg)
    return freqs, psd

def extract_band_power(freqs, psd):
    """
    Calculates the absolute power in different frequency bands given the PSD.
    """
    bands = {
        'Delta (0.5-4 Hz)': (0.5, 4.0),
        'Theta (4-8 Hz)': (4.0, 8.0),
        'Alpha (8-13 Hz)': (8.0, 13.0),
        'Beta (13-30 Hz)': (13.0, 30.0),
        'Gamma (30-45 Hz)': (30.0, 45.0)
    }
    
    band_powers = {}
    total_power = np.sum(psd)
    
    for band_name, (low, high) in bands.items():
        # Find intersecting frequencies
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.sum(psd[idx])
        band_powers[band_name] = band_power
        
    return band_powers

def detect_seizure_mock(data, sfreq):
    """
    A placeholder function for seizure detection.
    In a real scenario, this would load an ML model and run inference.
    """
    # Mock logic: if the standard deviation is unusually high, flag as "High Risk"
    std = np.std(data)
    if std > 50: # Arbitrary threshold for raw EEG in microvolts
        return "High Risk (Mock)"
    return "Normal (Mock)"
