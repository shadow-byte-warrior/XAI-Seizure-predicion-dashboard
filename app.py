import streamlit as st
import numpy as np
import pandas as pd
import time
import os
from scipy import signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import mne
from mne.viz import plot_topomap

from src.data_loader import load_edf, align_channels
from src.processing import preprocess_signal, normalize_signal, extract_bands, ensure_montage
from src.features import extract_features, compute_psd, extract_band_power
from src.visualization import plot_signal, plot_psd, plot_bands, plot_band_power_bar

from src.ml_pipeline import prepare_sequences, run_inference, load_seizure_intervals, is_seizure, MODEL_CONFIGS
from src.database import init_db, save_analysis, get_history, set_setting, get_setting

# Initialize Database
init_db()

# Set Streamlit page config
st.set_page_config(
    page_title="XAI SEIZURE PREDICTION APP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Management
current_theme = get_setting("theme", "dark")
if "theme_toggle" not in st.session_state:
    st.session_state.theme_toggle = current_theme

# Custom CSS for a Premium, Modern Look (Adaptive Glassmorphism)
theme_colors = {
    "dark": {
        "primary": "#818cf8",
        "secondary": "#c084fc",
        "accent": "#22d3ee",
        "bg": "radial-gradient(circle at top right, #1e1b4b, #0f172a)",
        "card_bg": "rgba(30, 41, 59, 0.4)",
        "text": "#f1f5f9",
        "text_dim": "#94a3b8",
        "border": "rgba(255, 255, 255, 0.08)",
        "glow": "0 0 20px rgba(129, 140, 248, 0.2)"
    },
    "light": {
        "primary": "#4f46e5",
        "secondary": "#9333ea",
        "accent": "#0891b2",
        "bg": "#f8fafc",
        "card_bg": "rgba(255, 255, 255, 0.8)",
        "text": "#1e293b",
        "text_dim": "#64748b",
        "border": "rgba(0, 0, 0, 0.05)",
        "glow": "0 0 20px rgba(79, 70, 229, 0.1)"
    }
}

colors = theme_colors[st.session_state.theme_toggle]

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Inter:wght@400;700&display=swap');
    
    :root {{
        --primary: {colors['primary']};
        --secondary: {colors['secondary']};
        --accent: {colors['accent']};
        --bg-dark: {colors['bg']};
        --card-bg: {colors['card_bg']};
        --text-main: {colors['text']};
        --text-dim: {colors['text_dim']};
        --border: {colors['border']};
        --glow: {colors['glow']};
    }}

    .stApp {{
        background: var(--bg-dark);
        color: var(--text-main);
        font-family: 'Outfit', sans-serif;
    }}
    
    h1, h2, h3 {{
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(to right, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }}

    .stButton>button {{
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 14px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(129, 140, 248, 0.3);
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 25px rgba(129, 140, 248, 0.5);
    }}

    .metric-card {{
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        padding: 24px;
        border-radius: 20px;
        text-align: center;
        transition: all 0.4s ease;
        color: var(--text-main);
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        border-color: var(--primary);
        box-shadow: var(--glow);
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
        background-color: transparent;
        padding: 10px 0;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 10px;
        padding: 10px 20px;
        color: var(--text-dim);
        font-weight: 600;
        transition: all 0.3s ease;
    }}

    .stTabs [aria-selected="true"] {{
        color: var(--primary) !important;
        background: rgba(129, 140, 248, 0.1) !important;
    }}

    /* Sidebar enhancement */
    section[data-testid="stSidebar"] {{
        background-color: { 'rgba(15, 23, 42, 0.98)' if st.session_state.theme_toggle == 'dark' else '#ffffff' };
        border-right: 1px solid var(--border);
    }}
    
    .status-badge {{
        display: inline-block;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-top: 12px;
    }}
    
    .status-seizure {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
    .status-normal {{ background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }}
    
    .hero-section {{
        text-align: center;
        padding: 5rem 2rem;
        background: radial-gradient(circle at center, rgba(129, 140, 248, 0.08) 0%, transparent 70%);
        border-radius: 40px;
        border: 1px solid var(--border);
        margin-bottom: 3rem;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("🧠 XAI SEIZURE PREDICTION")
st.markdown("Advanced EEG analytics with SHAP-driven explainability and multi-model inference.")

# Sidebar Configuration
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload EDF File", type=['edf'])
seizures_file = st.sidebar.file_uploader("Upload .seizures File (Optional)", type=['seizures', 'txt'])

st.sidebar.header("🎨 UI Customization")
theme_mode = st.sidebar.toggle("Light Mode", value=(st.session_state.theme_toggle == "light"))
new_theme = "light" if theme_mode else "dark"
if new_theme != st.session_state.theme_toggle:
    st.session_state.theme_toggle = new_theme
    set_setting("theme", new_theme)
    st.rerun()

main_tabs = st.tabs(["🚀 Analysis Dashboard", "📜 Historical Reports"])

with main_tabs[0]:
    if uploaded_file is None:
        st.markdown("""
            <div class='hero-section'>
                <h1 style='font-size: 3.5rem; margin-bottom: 1rem;'>🧠 XAI SEIZURE PREDICTION</h1>
                <p style='font-size: 1.25rem; color: #94a3b8; max-width: 850px; margin: 0 auto 2.5rem; line-height: 1.6;'>
                    Advanced EEG seizure detection powered by Explainable AI (XAI). 
                    Upload clinical EDF data to visualize neural patterns and receive model-driven diagnostic support.
                </p>
                <div style='display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;'>
                    <div class='metric-card' style='min-width: 220px;'>
                        <h3 style='margin: 0; color: var(--primary);'>18 Channels</h3>
                        <p style='margin: 0; font-size: 0.9rem; color: var(--text-dim);'>Standardized Alignment</p>
                    </div>
                    <div class='metric-card' style='min-width: 220px;'>
                        <h3 style='margin: 0; color: var(--secondary);'>SHAP Engine</h3>
                        <p style='margin: 0; font-size: 0.9rem; color: var(--text-dim);'>Feature Importance</p>
                    </div>
                    <div class='metric-card' style='min-width: 220px;'>
                        <h3 style='margin: 0; color: var(--accent);'>Multi-Model</h3>
                        <p style='margin: 0; font-size: 0.9rem; color: var(--text-dim);'>Cross-Validated Inference</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📁 1. Upload Data")
            st.write("Drag and drop your EDF files into the sidebar. Optional `.seizures` files enable ground-truth validation.")
        with col2:
            st.markdown("### 🤖 2. Select Model")
            st.write("Choose from various architectures (SCT, GRL, Federated) optimized for different EEG signatures.")
        with col3:
            st.markdown("### 📊 3. Analyze Results")
            st.write("Explore signal dynamics, spectral power, and AI-driven predictions with full interpretability.")
            
        st.info("👈 Use the sidebar to upload your first EDF file and begin the analysis.")
    else:
        with st.spinner("Loading and Normalizing EDF Data..."):
            try:
                eeg_data = load_edf(uploaded_file)
                st.sidebar.success("File loaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                st.stop()
                
        st.sidebar.header("🤖 Model Selection")
        selected_model_name = st.sidebar.selectbox("Select ML Model", list(MODEL_CONFIGS.keys()))
        config = MODEL_CONFIGS[selected_model_name]
        
        # Map Channels based on model requirements
        st.sidebar.markdown(f"**Model Requirements:** {config['channels']} channels")
        aligned_data, valid_channels = align_channels(eeg_data['raw'], config['channels'])
        
        assert aligned_data.shape[0] == config['channels'], "Channel alignment failed!"
        
        st.sidebar.header("🎛️ Visualization Parameters")
        selected_channel = st.sidebar.selectbox("Select EEG Channel to View", valid_channels if len(valid_channels)>0 else eeg_data['ch_names'])
        ch_idx = valid_channels.index(selected_channel) if selected_channel in valid_channels else 0
        
        duration = eeg_data['duration']
        max_duration = min(duration, 60.0)
        
        time_range = st.sidebar.slider(
            "Zoom Window (For UI Visualization Only)",
            min_value=0.0,
            max_value=float(duration),
            value=(0.0, max_duration),
            step=1.0
        )
        
        sfreq = eeg_data['sfreq']
        start_sample = int(time_range[0] * sfreq)
        end_sample = int(time_range[1] * sfreq)
        
        # For UI Visualization
        raw_signal_zoomed = aligned_data[ch_idx, start_sample:end_sample]
        time_axis_zoomed = eeg_data['times'][start_sample:end_sample]
        
        st.sidebar.header("⚙️ Processing Options")
        apply_filter = st.sidebar.checkbox("Apply Bandpass Filter (0.5 - 45 Hz)", value=True)
        apply_norm = st.sidebar.checkbox("Apply Z-Score Normalization", value=False)

        st.sidebar.header("🎛️ Pipeline Parameters")
        window_sec = st.sidebar.slider("Window Size (seconds)", 1.0, 10.0, 4.0, 0.5)
        window_size = int(window_sec * sfreq)
        step_sec = st.sidebar.slider("Step Size (seconds)", 0.5, 5.0, 2.0, 0.5)
        
        st.sidebar.header("🔍 SHAP XAI Options")
        enable_shap = st.sidebar.checkbox("Enable SHAP Channel Selection", value=True)
        if enable_shap:
            top_k = st.sidebar.slider("Top-K Channels to Keep", min_value=1, max_value=len(valid_channels), value=min(5, len(valid_channels)))
        else:
            top_k = len(valid_channels)
        
        processed_signal_zoomed = raw_signal_zoomed.copy()
        if apply_filter:
            processed_signal_zoomed = preprocess_signal(processed_signal_zoomed, sfreq)
        if apply_norm:
            processed_signal_zoomed = normalize_signal(processed_signal_zoomed)

        # --- CENTRALIZED ML PIPELINE ---
        model_path = config["filename"]
        pipeline_ready = False
        if os.path.exists(model_path):
            with st.spinner(f"Running Inference & XAI Engine..."):
                try:
                    # 1. Pipeline Execution
                    sequences, seq_times = prepare_sequences(aligned_data, sfreq, config, window_size=window_size, step_sec=step_sec)
                    predictions, probabilities, input_shape, top_ch_indices, ch_importance, shap_vals, test_sample = run_inference(
                        sequences, model_path, config, shap_enabled=enable_shap, top_k=top_k
                    )
                    
                    # 2. Validation / Ground Truth
                    intervals = None
                    accuracy = None
                    if seizures_file is not None:
                        intervals = load_seizure_intervals(seizures_file.getbuffer())
                        if intervals and len(intervals) > 0:
                            correct_preds = sum(1 for t, pred in zip(seq_times, predictions) if pred == is_seizure(t, intervals))
                            accuracy = (correct_preds / len(predictions) * 100) if len(predictions) > 0 else 0.0
                    
                    # 3. Global Metrics
                    if len(probabilities) >= 5:
                        smoothed_probs = pd.Series(probabilities).rolling(window=5, center=True, min_periods=1).mean().values
                        risk_score = np.max(smoothed_probs)
                    else:
                        risk_score = np.max(probabilities) if len(probabilities) > 0 else 0.0
                    
                    top_ch_names = [valid_channels[int(i)] for i in top_ch_indices] if enable_shap and top_ch_indices is not None else []
                    
                    # 4. Report Generation
                    report_content = f"""# XAI SEIZURE PREDICTION - Clinical Report
Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}
Source: {uploaded_file.name} | Model: {selected_model_name}

## Diagnostic Summary
- **Status:** {"⚠️ SEIZURE" if risk_score >= 0.5 else "✅ NON SEIZURE"}
- **Confidence:** {risk_score*100:.2f}%
- **Top Biomarkers:** {', '.join(top_ch_names)}
- **Timeline:** {f"{np.min(seq_times[predictions==1]):.1f}s to {np.max(seq_times[predictions==1]):.1f}s" if np.any(predictions==1) else "No seizure detected"}
"""
                    pipeline_ready = True
                except Exception as e:
                    st.error(f"Pipeline Error: {e}")
        else:
            st.error(f"Model file `{model_path}` not found.")

        # --- DASHBOARD STAGES ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Signal Overview", 
            "🌊 Spectral Analysis", 
            "🧠 Prediction & XAI",
            "🧮 Advanced Biomarkers",
            "📋 Clinical Report"
        ])

        with tab1:
            st.subheader(f"Signal Overview: {selected_channel}")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.plotly_chart(plot_signal(time_axis_zoomed, raw_signal_zoomed, f"Raw {selected_channel}", color="#818cf8"), use_container_width=True)
            with col_t2:
                st.plotly_chart(plot_signal(time_axis_zoomed, processed_signal_zoomed, f"Processed {selected_channel}", color="#22d3ee"), use_container_width=True)
                
            features = extract_features(processed_signal_zoomed, sfreq)
            cols = st.columns(3)
            cols[0].markdown(f"<div class='metric-card'><h4>Amplitude Mean</h4><h2>{features['Mean']:.2f} µV</h2></div>", unsafe_allow_html=True)
            cols[1].markdown(f"<div class='metric-card'><h4>Signal Variance</h4><h2>{features['Variance']:.2f}</h2></div>", unsafe_allow_html=True)
            cols[2].markdown(f"<div class='metric-card'><h4>Root Mean Square</h4><h2>{features['Std Dev']:.2f}</h2></div>", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("EEG Activity Heatmap (Multi-Channel Energy)")
            with st.spinner("Generating spatial activity heatmap..."):
                window_size_hm = int(2.0 * sfreq)
                total_samples_hm = aligned_data.shape[1]
                num_windows_hm = total_samples_hm // window_size_hm
                if num_windows_hm > 1:
                    hm_data = aligned_data[:, :num_windows_hm*window_size_hm].reshape(aligned_data.shape[0], num_windows_hm, window_size_hm)
                    energy_hm = np.log10(np.var(hm_data, axis=2) + 1e-7)
                    fig_hm = px.imshow(energy_hm, labels=dict(x="Time Steps", y="Channels", color="Log Var"), y=valid_channels, aspect="auto", color_continuous_scale="Plasma", template="plotly_dark")
                    fig_hm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_hm, use_container_width=True)
                else:
                    st.warning("Insufficient data duration for heatmap.")

        with tab2:
            st.subheader("Spectral Analysis & Topography")
            st_col1, st_col2 = st.columns([1.2, 0.8])
            
            with st_col1:
                st.markdown("#### Power Spectral Density (PSD)")
                freqs, psd = compute_psd(processed_signal_zoomed, sfreq)
                st.plotly_chart(plot_psd(freqs, psd, title=f"PSD for {selected_channel}"), use_container_width=True)
                st.plotly_chart(plot_band_power_bar(extract_band_power(freqs, psd)), use_container_width=True)

            with st_col2:
                st.markdown("#### Frequency Bands")
                with st.spinner("Extracting bands..."):
                    bands_dict = extract_bands(processed_signal_zoomed, sfreq)
                    st.plotly_chart(plot_bands(time_axis_zoomed, bands_dict), use_container_width=True)

            st.markdown("---")
            st.subheader("🗺️ Brain Topography (Spectral Power Distribution)")
            with st.spinner("Generating topomaps..."):
                try:
                    raw_topo, _ = ensure_montage(eeg_data['raw'].copy())
                    info = raw_topo.info
                    chs_with_pos = [ch['ch_name'] for ch in info['chs'] if (ch['loc'][0] != 0 or ch['loc'][1] != 0) and "_" not in ch['ch_name']]
                    if len(chs_with_pos) >= 4:
                        raw_topo.pick_channels(chs_with_pos)
                        t_mid = (time_range[0] + time_range[1]) / 2
                        data_topo, _ = raw_topo[:, max(0, int((t_mid-15)*sfreq)):min(int(duration*sfreq), int((t_mid+15)*sfreq))]
                        topo_bands = {'Delta': (0.5, 4.0), 'Theta': (4.0, 8.0), 'Alpha': (8.0, 13.0), 'Beta': (13.0, 30.0)}
                        t_cols = st.columns(4)
                        for idx, (b_name, (low, high)) in enumerate(topo_bands.items()):
                            powers = []
                            for ch in range(data_topo.shape[0]):
                                f, p = signal.welch(data_topo[ch], sfreq, nperseg=int(2*sfreq))
                                powers.append(np.mean(p[np.logical_and(f >= low, f <= high)]))
                            fig_t, ax_t = plt.subplots(figsize=(3, 3))
                            plot_topomap(np.array(powers), raw_topo.info, axes=ax_t, show=False, cmap='RdBu_r', contours=6)
                            ax_t.set_title(b_name, color='white', fontsize=10)
                            fig_t.patch.set_facecolor('none')
                            t_cols[idx].pyplot(fig_t)
                            plt.close(fig_t)
                    else:
                        st.warning("Insufficient electrode metadata for topography.")
                except Exception as e_topo:
                    st.error(f"Topomap Error: {e_topo}")

        with tab3:
            if not pipeline_ready:
                st.warning("Please ensure a valid model is selected and pipeline has completed.")
            else:
                st.subheader(f"AI Prediction: {selected_model_name}")
                st.success("Inference & XAI completed successfully!")
                
                st.markdown("---")
                res_col1, res_col2 = st.columns([1, 1])
                with res_col1:
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_score * 100,
                        title = {'text': "Max Seizure Risk %", 'font': {'size': 20}},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#818cf8"},
                            'steps': [
                                {'range': [0, 40], 'color': 'rgba(34, 197, 94, 0.1)'},
                                {'range': [40, 75], 'color': 'rgba(234, 179, 8, 0.1)'},
                                {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
                            ],
                            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
                        }
                    ))
                    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                with res_col2:
                    st.markdown("### 📊 Diagnostics & Validation")
                    diag_cols = st.columns(2)
                    diag_cols[0].markdown(f"<div class='metric-card'><h4>Segments</h4><h2>{len(sequences)}</h2></div>", unsafe_allow_html=True)
                    diag_cols[1].markdown(f"<div class='metric-card'><h4>Detections</h4><h2>{int(np.sum(predictions))}</h2></div>", unsafe_allow_html=True)
                    if accuracy is not None:
                        st.markdown(f"<div class='metric-card' style='margin-top:10px;'><h4>Accuracy</h4><h2>{accuracy:.2f}%</h2></div>", unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("🐝 SHAP Interpretability")
                if enable_shap and ch_importance is not None:
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        df_imp = pd.DataFrame({'Channel': valid_channels[:len(ch_importance)], 'Importance': ch_importance}).sort_values(by='Importance', ascending=False)
                        fig_imp = px.bar(df_imp, x='Channel', y='Importance', title="Global SHAP Importance", template="plotly_dark", color='Importance')
                        st.plotly_chart(fig_imp, use_container_width=True)
                    with col_s2:
                        if shap_vals is not None:
                            sv_agg = np.mean(shap_vals, axis=1)
                            data_agg = np.mean(test_sample.numpy() if hasattr(test_sample, 'numpy') else test_sample, axis=1)
                            df_shap = pd.DataFrame([{"Channel": valid_channels[i], "SHAP": sv_agg[b, i], "Signal": data_agg[b, i]} for i in range(len(valid_channels)) for b in range(sv_agg.shape[0])])
                            fig_bee = px.scatter(df_shap, x="SHAP", y="Channel", color="Signal", title="Impact on Risk", color_continuous_scale="RdBu_r", template="plotly_dark")
                            fig_bee.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_bee, use_container_width=True)
                else:
                    st.info("SHAP is disabled or failed to compute.")

                st.markdown("---")
                st.subheader("📈 Prediction Timeline")
                smoothed_probs_list = pd.Series(probabilities).rolling(window=5, center=True, min_periods=1).mean().tolist()
                fig_timeline = go.Figure(go.Bar(x=seq_times.tolist(), y=smoothed_probs_list, marker_color=['#f87171' if p == 1 else '#4ade80' for p in predictions]))
                fig_timeline.update_layout(title="Seizure Probability over Time", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
                st.plotly_chart(fig_timeline, use_container_width=True)

        with tab4:
            st.subheader("🧮 Advanced Biomarker Analysis")
            st.markdown("Extract and analyze custom statistical and spectral features to identify clinical biomarkers.")
            available_feats = ['Mean', 'Variance', 'Std Dev', 'Skewness', 'Kurtosis', 'Entropy', 'RMS', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
            selected_feats = st.multiselect("Select Features for Analysis", available_feats, default=['Variance', 'Skewness', 'Entropy', 'Alpha', 'Beta'])
            
            if st.button("🚀 Run Multi-Feature Analysis"):
                with st.spinner("Extracting features..."):
                    try:
                        feat_list = []
                        subsample_step = max(1, len(sequences) // 250) 
                        for i in range(0, len(sequences), subsample_step):
                            s_start = max(0, int((seq_times[i] - window_sec/2) * sfreq))
                            s_end = min(aligned_data.shape[1], int((seq_times[i] + window_sec/2) * sfreq))
                            seg_raw = aligned_data[ch_idx, s_start:s_end]
                            if len(seg_raw) < 100: continue
                            f_dict = extract_features(seg_raw, sfreq)
                            freqs_s, psd_s = compute_psd(seg_raw, sfreq)
                            b_dict = {k.split(' ')[0]: v for k, v in extract_band_power(freqs_s, psd_s).items()}
                            full_dict = {**f_dict, **b_dict}
                            label = 'Normal'
                            if intervals and is_seizure(seq_times[i], intervals) == 1:
                                label = 'Seizure'
                            
                            feat_dict = {**{k: full_dict[k] for k in selected_feats if k in full_dict}, 'Time': seq_times[i], 'Label': label}
                            feat_list.append(feat_dict)
                        df_feats = pd.DataFrame(feat_list)
                        if not df_feats.empty:
                            col_f1, col_f2 = st.columns(2)
                            with col_f1:
                                st.markdown("#### Correlation Matrix")
                                st.plotly_chart(px.imshow(df_feats.drop(columns=['Time', 'Label']).corr(), text_auto=".2f", color_continuous_scale='RdBu_r', template='plotly_dark'), use_container_width=True)
                            with col_f2:
                                st.markdown("#### Feature Importance Beeswarm")
                                df_melt = df_feats.melt(id_vars=['Label'], value_vars=selected_feats)
                                st.plotly_chart(px.strip(df_melt, x="value", y="variable", color="Label", stripmode="overlay", template='plotly_dark', color_discrete_map={'Seizure': '#f87171', 'Normal': '#4ade80'}), use_container_width=True)
                        else:
                            st.warning("No features extracted.")
                    except Exception as feat_err:
                        st.error(f"Feature Analysis Error: {feat_err}")

        with tab5:
            if not pipeline_ready:
                st.info("Run the prediction pipeline first.")
            else:
                st.subheader("📋 Final Clinical Assessment")
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    is_seizure_detected = risk_score >= 0.5
                    color = "#f87171" if is_seizure_detected else "#4ade80"
                    st.markdown(f"<div style='border: 2px solid {color}; padding: 30px; border-radius: 20px; text-align: center; background: rgba(0,0,0,0.2);'><h3 style='color: var(--text-dim); margin:0;'>Assessment</h3><h1 style='color: {color}; font-size: 3rem; margin: 10px 0;'>{'SEIZURE' if is_seizure_detected else 'NORMAL'}</h1><p>Risk: {risk_score*100:.1f}%</p></div>", unsafe_allow_html=True)
                with res_col2:
                    st.markdown("#### Evidence Summary")
                    st.write(f"- Identified {int(np.sum(predictions))} segments of rhythmic ictal activity.")
                    st.write(f"- Critical Timeline: {f'{np.min(seq_times[predictions==1]):.1f}s to {np.max(seq_times[predictions==1]):.1f}s' if np.any(predictions==1) else 'None'}")
                    st.write(f"- Primary Biomarkers: {', '.join(top_ch_names)}")
                
                st.markdown("---")
                st.subheader("📥 Export & History")
                save_analysis(filename=uploaded_file.name, model_name=selected_model_name, risk_score=float(risk_score), detections=int(np.sum(predictions)), total_segments=len(sequences), accuracy=float(accuracy) if accuracy else None, top_channels=top_ch_names, report_content=report_content)
                st.download_button("📥 Download MD Report", report_content, file_name=f"report_{int(time.time())}.md", use_container_width=True)
                try:
                    from src.report import generate_pdf_report
                    pdf_bytes = generate_pdf_report({'filename': uploaded_file.name, 'model_name': selected_model_name, 'risk_score': float(risk_score), 'total_segments': len(sequences), 'detections': int(np.sum(predictions)), 'accuracy': f"{accuracy:.2f}%" if accuracy else "N/A", 'top_channels': top_ch_names, 'clinical_notes': "Seizure activity detected." if is_seizure_detected else "Normal EEG."})
                    st.download_button("📄 Download PDF Report", bytes(pdf_bytes), file_name=f"report_{int(time.time())}.pdf", mime="application/pdf", use_container_width=True)
                except Exception as pdf_err:
                    st.error(f"PDF Error: {pdf_err}")

with main_tabs[1]:
    st.subheader("📜 Recent Analysis History")
    history = get_history()
    if not history:
        st.info("No analysis history found. Run your first prediction to see it here!")
    else:
        for item in history:
            with st.expander(f"📅 {item['timestamp']} | {item['filename']} | {item['model_name']}"):
                h_col1, h_col2 = st.columns([2, 1])
                with h_col1:
                    st.markdown(f"**Max Risk:** `{item['risk_score']*100:.2f}%` | **Detections:** `{item['detections']}`")
                    st.markdown(f"**Channels:** `{', '.join(item['top_channels'])}`")
                with h_col2:
                    st.download_button(
                        "📥 Download Report", 
                        item['report_content'], 
                        file_name=f"diagnostic_report_{item['id']}.md",
                        key=f"dl_{item['id']}"
                    )
                if st.button(f"🗑️ Delete Entry", key=f"del_{item['id']}"):
                    from src.database import delete_analysis
                    delete_analysis(item['id'])
                    st.rerun()
