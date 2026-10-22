# Experimental Protocol Specification

## Evaluation Protocol: Leave-One-Subject-Out (LOSO) Cross-Validation
- For each outer fold $k \in \{1, \dots, S\}$:
  1. Participant $k$ is held out as the test set.
  2. Remaining $S-1$ participants are partitioned into training and internal validation subjects.
  3. Preprocessing scalers (mean, std) and reliability baseline statistics are fitted strictly on the training subjects.
  4. Models (Random Forest and CNN-LSTM) are trained on the training fold.
  5. Internal validation subjects are used for CNN-LSTM early stopping.
  6. The held-out test sequence is evaluated chronologically.
  7. No future information, test subject data, or ground-truth fault masks are ever exposed to the models or controller.
