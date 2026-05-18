import torch
import numpy as np
import warnings

# Suppress some noisy warnings for cleaner test output
warnings.filterwarnings("ignore")

print("=========================================")
print("[Testing] NeuroGuard: Multi-Model Validation Test")
print("=========================================")

model_files = [
    "dagrl_model.pt",
    "dasct_model.pt",
    "grl_model.pt",
    "sct_model.pt",
    "federated_model.pt",
    "model_best_xai_combined.pt"
]

for model_path in model_files:
    print(f"\n[{model_path}]")
    
    # 1. Test Load
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        print(f"  [OK] Load Success: type={type(model)}")
        if isinstance(model, dict):
            print(f"  [DICT KEYS]: {list(model.keys())}")
        elif hasattr(model, 'eval'):
            model.eval()
            print("  [OK] Successfully called model.eval()")
    except Exception as e:
        print(f"  [FAIL] Load Failed: {e}")
        continue

print("\n=========================================")
print("Tests Completed.")
print("=========================================")

