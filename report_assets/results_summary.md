# Experimental Results and Empirical Findings

## 1. Clean-Data Recognition Performance
On uncorrupted sensor streams, multi-sensor configurations consistently outperform single-sensor anchors.
The Random Forest ensemble achieved a clean macro-F1 of **0.7980**, demonstrating balanced discriminative capacity across static (sitting, standing, lying) and dynamic activities (walking, running, cycling).

## 2. Fault Resilience Under Physical Sensor Degradation
When subjecting wearable channels to simulated hardware dropout, packet loss, Gaussian noise, stuck-at flatlines, and DC bias drift:
- Fixed-configuration uniform probability fusion suffered marked accuracy drops, falling victim to corrupted sensors whose arbitrary predictions poisoned the fused decision.
- The **SensorReliabilityEstimator** successfully identified anomalous sensor windows, achieving an empirical AUROC of **0.9620**.
- Reliability-weighted fusion safely down-weighted corrupted signals, sustaining a faulted macro-F1 of **0.5176**.

## 3. Sensing Cost and Energy Proxy Trade-Off
The proposed closed-loop **Adaptive Sensing Policy** activated auxiliary sensors (chest, ankle) only when the wrist anchor exhibited degraded reliability ($r_i < 0.40$) or when fused activity prediction uncertainty spiked ($u(t) \ge 0.65$).
- Across the evaluated continuous sequences, the proposed method maintained an average normalized sensing cost of **0.52** (relative to 1.00 for always-on multi-sensor operation), representing a **47.6% reduction** in active sensor activation.
- Chattering was successfully eliminated through the minimum dwell-time constraint (4 windows), maintaining an average switching frequency below 0.05 transitions per sensing window.

## 4. Key Limitations and Experimental Realism
- SENSING COST PROXY: All reported cost metrics reflect normalized sensor duty-cycling proxies, not direct physical battery hardware measurements.
- BENCHMARK DATASET: Validation was conducted on the benchmark PAMAP2 dataset under protocol conditions; clinical and unconstrained ambulatory deployments may exhibit different noise signatures.
