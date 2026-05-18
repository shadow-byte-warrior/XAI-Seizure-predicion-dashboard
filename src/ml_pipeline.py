import torch
import numpy as np
import re
import shap
from src.processing import butter_bandpass_filter, preprocess_signal, normalize_signal
from src.models import SCT, GRLSeizureModel, FederatedModel, DASCT, DAGRLModel

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() == 1:
            out = out.unsqueeze(1)
        return out

MODEL_CONFIGS = {
    "DA-GRL": {
        "filename": "models/dagrl_model.pt",
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": True,
        "class": DAGRLModel
    },
    "DA-SCT": {
        "filename": "models/dasct_model.pt",
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": True,
        "class": DASCT
    },
    "GRL": {
        "filename": "models/grl_model.pt",
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": True,
        "class": GRLSeizureModel
    },
    "SCT": {
        "filename": "models/sct_model.pt",
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": False,
        "class": SCT
    },
    "Federated": {
        "filename": "models/federated_model.pt",
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": True,
        "class": FederatedModel
    },
    "Model combined": {
        "filename": "models/model_best_xai_combined.pt",
        "channels": 18,
        "features_per_channel": 5,
        "requires_sequence": True,
        "class": None # Dynamic
    }
}

def extract_segment_features(segment, sfreq):
    """
    Extracts band powers (Delta, Theta, Alpha, Beta, Gamma) for all channels in a 2-second segment.
    Applies Z-score normalization per channel.
    segment shape: [num_channels, window_size]
    Returns: flattened feature array of size [num_channels * 5]
    """
    bands = {
        'Delta': (0.5, 4.0),
        'Theta': (4.0, 8.0),
        'Alpha': (8.0, 13.0),
        'Beta': (13.0, 30.0),
        'Gamma': (30.0, 45.0)
    }
    
    num_channels = segment.shape[0]
    segment_features = []
    
    for ch in range(num_channels):
        ch_data = segment[ch, :]
        
        # Z-score normalization
        mean = np.mean(ch_data)
        std = np.std(ch_data)
        if std > 0:
            ch_data = (ch_data - mean) / std
            
        ch_powers = []
        for band_name, (low, high) in bands.items():
            filtered = butter_bandpass_filter(ch_data, low, high, sfreq, order=3)
            # Absolute power
            power = np.sum(filtered ** 2) / len(filtered)
            
            # Robust Feature Scaling: Log10 Transformation (Decibels)
            # Compresses extreme dynamic ranges (high amplitude artifacts)
            log_power = 10 * np.log10(power + 1e-9)
            ch_powers.append(log_power)
            
        segment_features.extend(ch_powers)
        
    segment_features = np.array(segment_features)
    
    # Feature Standardization (Z-score normalize the *features* themselves)
    # This guarantees the model receives inputs centered at 0 with a std of 1
    mean_feat = np.mean(segment_features)
    std_feat = np.std(segment_features)
    if std_feat > 0:
        segment_features = (segment_features - mean_feat) / std_feat
        
    return segment_features

def prepare_sequences(data, sfreq, config, window_size=1024, step_sec=2.0):
    """
    Splits data into windows and returns preprocessed sequences.
    Optimized: Preprocesses the entire signal first.
    """
    num_channels, total_samples = data.shape
    
    # 1. Preprocess the entire signal at once
    processed_data = np.zeros_like(data)
    for ch in range(num_channels):
        ch_data = preprocess_signal(data[ch], sfreq, lowcut=0.5, highcut=100.0)
        processed_data[ch] = normalize_signal(ch_data)
    
    sequences = []
    sequence_times = []
    
    # 2. Slice into windows
    step = int(step_sec * sfreq)
    
    for i in range(0, total_samples - window_size + 1, step):
        segment = processed_data[:, i:i+window_size]
        sequences.append(segment.T)
        sequence_times.append((i + window_size/2) / sfreq)
            
    return np.array(sequences), np.array(sequence_times)

def compute_shap_importance(model, sequences, config, top_k=5):
    """
    Computes SHAP values to identify Top-K important channels.
    Wraps the model to handle tuple outputs.
    """
    model.eval()
    wrapper = ModelWrapper(model)
    
    num_channels = config["channels"]
    
    # Select a diverse background for SHAP (approximating balanced background)
    bg_size = min(50, len(sequences))
    background = torch.tensor(sequences[:bg_size]).float()
    
    # Ensure test samples are representative
    test_sample = torch.tensor(sequences[::max(1, len(sequences)//10)][:10]).float()
        
    try:
        explainer = shap.DeepExplainer(wrapper, background)
        shap_values = explainer.shap_values(test_sample)
    except Exception as e:
        try:
            explainer = shap.GradientExplainer(wrapper, background)
            shap_values = explainer.shap_values(test_sample)
        except Exception as e2:
            print(f"[ERROR] SHAP explainer failed: {e2}")
            return None, None

    if isinstance(shap_values, list):
        # shap returns list for multiple outputs. 
        # If we have (batch, 1) output, it might return list of length 1 or 2 (if class probability)
        shap_vals = shap_values[0] if len(shap_values) > 0 else shap_values
    else:
        shap_vals = shap_values
        
    # Aggregate SHAP to Channel Level
    # shap_vals shape: (batch, T, channels)
    if shap_vals.ndim == 4: # Handle (batch, T, channels, 1)
        shap_vals = shap_vals.squeeze(-1)
        
    importance = np.abs(shap_vals).mean(axis=(0, 1))
    
    # Select Top Channels
    selected_channels = [int(np.asarray(i).item()) for i in np.argsort(importance)[-top_k:][::-1]]
    
    return selected_channels, importance, shap_vals, test_sample

def apply_channel_mask(sequences, selected_channels, config):
    """
    Zeroes out the non-selected channels to maintain model input shape.
    """
    if selected_channels is None:
        return sequences
        
    masked_sequences = sequences.copy()
    num_channels = config["channels"]
    
    all_indices = np.arange(num_channels)
    to_mask = [i for i in all_indices if i not in selected_channels]
    
    # sequences shape: (batch, T, channels)
    masked_sequences[:, :, to_mask] = 0.0
    
    return masked_sequences

def run_inference(sequences, model_path, config, shap_enabled=False, top_k=5):
    """
    Loads model and runs inference using batching.
    """
    try:
        loaded = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        if isinstance(loaded, dict):
            # 1. Determine Class
            model_class = config.get("class")
            
            # Handle combined model edge-case
            if model_class is None and "best_model_type" in loaded:
                m_type = loaded["best_model_type"]
                if m_type == "dagrl": model_class = DAGRLModel
                elif m_type == "dasct": model_class = DASCT
                elif m_type == "grl": model_class = GRLSeizureModel
                elif m_type == "sct": model_class = SCT
                elif m_type == "federated": model_class = FederatedModel
            
            if model_class is None:
                raise ValueError(f"Could not determine model class for {model_path}")
                
            # 2. Instantiate and Load
            n_ch = config["channels"]
            model = model_class(n_ch=n_ch)
            model.load_state_dict(loaded["model_state_dict"])
        else:
            model = loaded
            
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load/initialize model: {e}")

    selected_channels = None
    channel_importance = None
    shap_vals = None
    test_sample = None
    
    if shap_enabled:
        selected_channels, channel_importance, shap_vals, test_sample = compute_shap_importance(model, sequences, config, top_k)

    if len(sequences) == 0:
        return np.array([]), np.array([]), (0,0,0), None, None, None, None

    # Batch Inference
    X = torch.tensor(sequences).float()
    
    with torch.no_grad():
        outputs = model(X)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        probs = outputs.squeeze().numpy()
        
    if probs.ndim == 0:
        probs = np.array([probs])
        
    predictions = (probs > 0.5).astype(int)
    sample_shape = tuple(X[0].unsqueeze(0).shape)
            
    return predictions, probs, sample_shape, selected_channels, channel_importance, shap_vals, test_sample

def load_seizure_intervals(file_content):
    try:
        text = file_content.decode("utf-8")
    except UnicodeDecodeError:
        print("[WARNING] .seizures file cannot be decoded as UTF-8.")
        return None
        
    starts = re.findall(r'Start Time:\s*(\d+)', text)
    ends = re.findall(r'End Time:\s*(\d+)', text)
    
    if not starts or not ends or len(starts) != len(ends):
        print(f"[WARNING] No valid intervals found in .seizures file. (Starts: {len(starts)}, Ends: {len(ends)})")
        return None
        
    intervals = []
    for s, e in zip(starts, ends):
        intervals.append((int(s), int(e)))
        
    return intervals

def is_seizure(time_sec, intervals):
    for start, end in intervals:
        if start <= time_sec <= end:
            return 1
    return 0
