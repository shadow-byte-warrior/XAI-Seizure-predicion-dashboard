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
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            if 'model_name' in data:
                print(f"Model Name: {data['model_name']}")
            if 'best_model_type' in data:
                print(f"Best Model Type: {data['best_model_type']}")
        else:
            print(f"Object: {data}")
    except Exception as e:
        print(f"Error: {e}")
