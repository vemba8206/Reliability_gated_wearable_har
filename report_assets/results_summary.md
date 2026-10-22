# Experimental Results and Empirical Findings

## 1. Clean-Data Recognition Performance
The proposed Random Forest ensemble achieved a clean macro-F1 of **0.8574** on the configured evaluation split. Compare its baseline rows before claiming superiority.

## 2. Fault Resilience Under Physical Sensor Degradation
When subjecting wearable channels to simulated hardware dropout, packet loss, Gaussian noise, stuck-at flatlines, and DC bias drift:
- Mean fault-detection AUROC across configured scenarios and folds was **0.8882**, comparing clean and injected-fault windows.
- Proposed macro-F1 under moderate dropout was **0.7919**. Other fault conditions may be worse; inspect the comparison table.

## 3. Sensing Cost and Energy Proxy Trade-Off
The proposed closed-loop **Adaptive Sensing Policy** activated auxiliary sensors in response to degraded reliability, high uncertainty, or periodic probing, subject to the configured budget.
- Across the evaluated continuous sequences, the proposed method maintained an average normalized sensing cost of **0.53** (relative to 1.00 for always-on multi-sensor operation), representing a **47.2% reduction** in active sensor activation.
- Switching behavior is reported in the cost comparison table; a dwell constraint does not guarantee zero oscillation.

## 4. Key Limitations and Experimental Realism
- SENSING COST PROXY: All reported cost metrics reflect normalized sensor duty-cycling proxies, not direct physical battery hardware measurements.
- BENCHMARK DATASET: Validation was conducted on the benchmark PAMAP2 dataset under protocol conditions; clinical and unconstrained ambulatory deployments may exhibit different noise signatures.
