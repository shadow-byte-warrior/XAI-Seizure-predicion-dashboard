# 🧠 NeuroGuard: System Architecture

NeuroGuard is a high-integrity EEG signal processing and seizure detection platform. It combines state-of-the-art Deep Learning models with Explainable AI (XAI) and a modern, high-performance UI.

![NeuroGuard Architecture Diagram](file:///C:/Users/Rishi/.gemini/antigravity/brain/e55acb39-2ebd-4a42-bc63-41a9629d296e/neuroguard_architecture_1777603460269.png)

## 🏗️ Core Components

### 1. Data Ingestion & Alignment
- **Source:** Clinical EDF (European Data Format) files.
- **Normalization:** Standardizes channel names (e.g., `FP1-F7`) and aligns them to the model's required input shape (18 or 23 channels).
- **Fallback:** Missing channels are automatically filled with zeros to maintain pipeline integrity.

### 2. Optimized Preprocessing Pipeline
- **Vectorized Filtering:** Applies Butterworth bandpass filters (0.5 - 100 Hz) and Z-score normalization globally across the entire signal for maximum efficiency.
- **Dynamic Windowing:** Splits the signal into overlapping segments (default 4s windows with 50% overlap). Parameters are fully configurable in the UI.

### 3. Multi-Model Inference Engine
NeuroGuard supports several advanced architectures:
- **DA-GRL / GRL:** Domain-Adversarial Gradient Reversal Layer models for robust feature extraction.
- **DA-SCT / SCT:** Spiking ConvNet-Transformer models for efficient temporal analysis.
- **Federated:** Models optimized for decentralized data learning.

### 4. Explainable AI (XAI)
- **SHAP DeepExplainer:** Uses model-specific gradient analysis to identify the top-K channels contributing to seizure detection.
- **Automatic Channel Selection:** Zeroes out non-contributing channels to verify prediction stability.

### 5. Persistence & Reporting
- **SQL Backend:** SQLite database (`neuroguard.db`) stores every analysis session, allowing for historical review and clinical report retrieval.
- **Reporting:** Generates Markdown/PDF clinical reports with detailed statistics and XAI findings.

### 6. Modern Analytics Dashboard
- **Responsive UI:** Built with Streamlit, featuring a premium Dark/Light mode system.
- **Visualizations:** High-performance Plotly charts for time-series, spectral density, and risk timelines.

## 🛠️ Technology Stack
- **Deep Learning:** PyTorch
- **Signal Processing:** MNE-Python, SciPy
- **XAI:** SHAP
- **Database:** SQLite3
- **Frontend:** Streamlit, Plotly
