# IEEE Figure Captions and Discussion

- **Figure 1 (`fig1_activity_distribution.png`)**: Class distribution of usable protocol windows across the 6 target activities (Lying, Sitting, Standing, Walking, Running, Cycling).
- **Figure 2 (`fig2_clean_confusion_matrix_rf.png`)**: Confusion matrix for the Sensor-Wise Random Forest ensemble on clean test subjects under LOSO cross-validation.
- **Figure 3 (`fig3_clean_confusion_matrix_cnn.png`)**: Confusion matrix for the Sensor-Wise CNN-LSTM hybrid model on clean test subjects.
- **Figure 4 (`fig4_faulted_confusion_matrix.png`)**: Confusion matrix under severe sensor fault injection demonstrating resilience of reliability-gated fusion.
- **Figure 5 (`fig5_macro_f1_vs_severity.png`)**: Macro-F1 recognition accuracy as a function of fault corruption severity (0.0 to 1.0) comparing the five baselines.
- **Figure 6 (`fig6_fault_type_comparison.png`)**: Robustness profile across different fault modes: dropout, intermittent loss, Gaussian noise, flatline, and drift.
- **Figure 7 (`fig7_cost_vs_macro_f1.png`)**: Pareto trade-off between normalized sensing cost proxy $C(t)$ and faulted Macro-F1 score.
- **Figure 8 (`fig8_active_sensor_ratio.png`)**: Mean active-sensor ratio comparing static and adaptive sensing strategies.
- **Figure 9 (`fig9_reliability_timeline.png`)**: Chronological trajectory of estimated sensor reliability scores $r_i(t)$ following fault injection.
- **Figure 10 (`fig10_activation_timeline.png`)**: Closed-loop sensor activation/deactivation decisions over a representative 100-window test sequence.
- **Figure 11 (`fig11_cnn_training_curves.png`)**: PyTorch CNN-LSTM training and validation loss progression with early stopping.
- **Figure 12 (`fig12_rf_feature_importance.png`)**: Top Gini feature importances for the Random Forest classifier.
- **Figure 13 (`fig13_inference_latency.png`)**: Per-window inference latency across the evaluated classifiers.
