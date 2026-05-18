import torch

model_files = [
    "dagrl_model.pt",
    "dasct_model.pt",
    "grl_model.pt",
    "sct_model.pt",
    "federated_model.pt",
    "model_best_xai_combined.pt"
]

for model_path in model_files:
    print(f"\n--- {model_path} ---")
    try:
        data = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            sd = data.get('model_state_dict', {})
            # Look for conv weight shapes
            for k, v in sd.items():
                if 'weight' in k and v.dim() >= 2:
                    # Usually conv1 in these models has shape (out, in, k)
                    # or linear has (out, in)
                    print(f"Key: {k}, Shape: {v.shape}")
                    break 
        else:
            print("Not a dict")
    except Exception as e:
        print(f"Error: {e}")
