import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

def plot_signal(time, data, title, color='#1f77b4', height=400):
    """
    Plots a time-series EEG signal using Plotly.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=data, mode='lines', name='Signal', line=dict(color=color, width=1)))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        height=height,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=30),
        hovermode="x unified"
    )
    return fig

def plot_psd(freqs, psd, title="Power Spectral Density"):
    """
    Plots the Power Spectral Density.
    """
    fig = go.Figure()
    
    # Usually, it's better to plot PSD on a logarithmic scale (10*log10)
    # But for simplicity, we'll plot raw or log linearly
    # Ensure no zero values for log10
    psd_db = 10 * np.log10(np.maximum(psd, 1e-15))
    
    fig.add_trace(go.Scatter(x=freqs, y=psd_db, mode='lines', name='PSD', line=dict(color='#ff7f0e')))
    
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power/Frequency (dB/Hz)",
        height=400,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis=dict(range=[0, 50]) # Focus on 0-50Hz range for standard EEG
    )
    return fig

def plot_bands(time, bands_dict):
    """
    Plots all extracted frequency bands in subplots or a unified view.
    We'll use separate traces in one figure with subplots via make_subplots if we want,
    but layering them or making a grid of figures in Streamlit is often easier.
    Here we return a single multi-trace figure.
    """
    from plotly.subplots import make_subplots
    
    num_bands = len(bands_dict)
    fig = make_subplots(rows=num_bands, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=list(bands_dict.keys()))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, ((band_name, band_data), color) in enumerate(zip(bands_dict.items(), colors)):
        fig.add_trace(
            go.Scatter(x=time, y=band_data, mode='lines', name=band_name, line=dict(color=color, width=1)),
            row=i+1, col=1
        )
        
    fig.update_layout(
        height=150 * num_bands,
        title="EEG Frequency Bands",
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=30)
    )
    
    # Update y-axis titles
    for i in range(num_bands):
        fig.layout[f'yaxis{i+1}'].title = "Amp"
        
    fig.layout['xaxis' + str(num_bands)].title = "Time (s)"
    
    return fig
    
def plot_band_power_bar(band_powers):
    """
    Plots a bar chart of the relative or absolute band powers.
    """
    df = pd.DataFrame(list(band_powers.items()), columns=['Band', 'Power'])
    
    fig = px.bar(df, x='Band', y='Power', title="Band Power Distribution",
                 color='Band', template="plotly_dark")
                 
    fig.update_layout(
        height=400,
        xaxis_title="Frequency Band",
        yaxis_title="Absolute Power",
        margin=dict(l=40, r=20, t=40, b=30)
    )
    return fig
