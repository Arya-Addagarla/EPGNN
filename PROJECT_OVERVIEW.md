# EPGNN Project Overview

## Objective
To develop a high-reliability Multimodal Graph Neural Network (EPGNN) capable of detecting pre-seismic indicators from massive, multivariate seismogram arrays.

## Datasets
- **Primary:** Stanford Earthquake Dataset (STEAD) - 80GB of high-resolution seismic waveforms and associated categorical metadata.

## Methodology
The framework treats individual seismic channels (E, N, Z) or distinct geographical sensors as nodes in a graph. 
1. **Temporal Extraction:** 1D-CNNs extract sequential embeddings from the raw waveform data.
2. **Spatial Correlation:** Graph Convolutional layers process the interactions between sensors to differentiate true tectonic shifts from localized noise.

## Success Metrics
- Overall accuracy > 90%
- F1-Score prioritization to minimize false-positive "crying wolf" scenarios
- Minimized Epicenter/Magnitude Mean Squared Error (MSE)
