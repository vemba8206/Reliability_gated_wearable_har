# Research Methodology Summary

## Pipeline Architecture
1. **Acquisition & Preprocessing**: PAMAP2 protocol recordings (100 Hz, 54 columns) loaded with exact channel mappings (Wrist cols 4-6, 10-12; Chest cols 21-23, 27-29; Ankle cols 38-40, 44-46).
2. **Leakage Prevention**: Subject-wise Leave-One-Subject-Out (LOSO) partitioning across all protocol participants. Preprocessing scaling parameters are strictly fitted on training participants.
3. **Model Families**:
   - Model A: Sensor-wise Random Forest (72 statistical features per window, probability alignment).
   - Model B: Sensor-wise CNN-LSTM (1D Conv temporal-spatial filters + LSTM sequence modeling).
4. **Reliability Estimation**: Computes observable quality indicators: missing-sample ratio, flatline variance, noise first-difference variance, and sensor clipping.
5. **Uncertainty & Fusion**: Reliability-weighted probability fusion with normalized Shannon entropy quantification.
6. **Closed-Loop Controller**: Regulates active sensor configuration for the NEXT sensing window $t+1$ under physical realism rules.
