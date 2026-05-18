import mne
import numpy as np
import matplotlib.pyplot as plt
from src.processing import ensure_montage
from scipy import signal
import sys

# Ensure UTF-8 output for Windows terminal emojis
sys.stdout.reconfigure(encoding='utf-8')

def test_topomap_logic(edf_path):
    print(f"Testing Topomap Engine on: {edf_path}")
    try:
        # Load data
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        print(f"Original Channels: {raw.ch_names[:5]}...")
        
        # Apply Robust Montage Engine
        raw_topo, msg = ensure_montage(raw.copy())
        if msg:
            print(f"Status: {msg}")
            # If we have a message but dig points were found anyway, continue
        else:
            print("Status: Success (Direct Match)")
            
        # Validation Layer
        info = raw_topo.info
        dig = info.get('dig')
        if dig is None:
            print("FAILED: info['dig'] is None")
            return
            
        chs_with_pos = [ch['ch_name'] for ch in info['chs'] if (ch['loc'][0] != 0 or ch['loc'][1] != 0) and "_" not in ch['ch_name']]
        print(f"Mapped Channels ({len(chs_with_pos)}): {chs_with_pos[:10]}...")
        
        if len(chs_with_pos) < 4:
            print(f"FAILED: Too few mapped channels ({len(chs_with_pos)}).")
            return

        # Pick mapped channels
        raw_topo.pick_channels(chs_with_pos)
        
        # Simulate power calculation (Delta band)
        data, _ = raw_topo[:, :int(10 * raw_topo.info['sfreq'])] # 10s
        powers = []
        sfreq = raw_topo.info['sfreq']
        for ch in range(data.shape[0]):
            f, p = signal.welch(data[ch], sfreq, nperseg=int(2*sfreq))
            idx_f = np.logical_and(f >= 0.5, f <= 4.0)
            powers.append(np.mean(p[idx_f]))
        
        powers = np.array(powers)
        
        # Plot Topomap (save to file)
        from mne.viz import plot_topomap
        fig, ax = plt.subplots()
        plot_topomap(powers, raw_topo.info, axes=ax, show=False)
        ax.set_title("Test Topomap (Delta)")
        fig.savefig("test_topomap_result.png")
        print("SUCCESS: Topomap generated and saved to test_topomap_result.png")
        
    except Exception as e:
        print(f"ERROR during test: {e}")

if __name__ == "__main__":
    # Test with a known file
    test_file = r"non seizure\chb01_01.edf"
    test_topomap_logic(test_file)
