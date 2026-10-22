"""Closed-loop adaptive sensing controller for fault-resilient wearable IoT."""

import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class AdaptiveSensingController:
    """
    Lightweight rule-based runtime controller deciding active sensor configuration
    for the NEXT sensing window (t+1) based on current reliability, prediction uncertainty,
    and normalized sensing cost proxy.
    
    Adheres strictly to experimental realism:
    - Inactive sensors are not read or evaluated until activated or probed.
    - Probing incurs active sensing cost for that window.
    - Switching decisions incorporate minimum dwell times to avoid chattering.
    - At least one usable sensor remains active at all times.
    """
    def __init__(
        self,
        sensors: Optional[List[str]] = None,
        anchor_sensor: str = "wrist",
        r_low: float = 0.40,
        r_good: float = 0.70,
        u_high: float = 0.65,
        u_low: float = 0.35,
        min_dwell_windows: int = 4,
        probe_interval: int = 20,
        cost_weights: Optional[Dict[str, float]] = None,
        cost_budget: float = 1.0
    ):
        if sensors is None:
            self.sensors = ["wrist", "chest", "ankle"]
        else:
            self.sensors = sensors

        self.anchor_sensor = anchor_sensor
        self.r_low = r_low
        self.r_good = r_good
        self.u_high = u_high
        self.u_low = u_low
        self.min_dwell_windows = min_dwell_windows
        self.probe_interval = probe_interval

        if cost_weights is None:
            self.cost_weights = {s: 1.0 for s in self.sensors}
        else:
            self.cost_weights = cost_weights

        self.total_cost_weight = sum(self.cost_weights.values())
        if not 0 < cost_budget <= 1 or self.cost_weights[anchor_sensor] / self.total_cost_weight > cost_budget:
            raise ValueError("Cost budget must include the anchor and lie in (0, 1].")
        self.cost_budget = cost_budget

        # Controller state variables
        self.active_mask: Dict[str, int] = {s: 1 if s == anchor_sensor else 0 for s in self.sensors}
        self.last_observed_reliability: Dict[str, float] = {s: 1.0 for s in self.sensors}
        self.windows_since_last_observed: Dict[str, int] = {s: 0 for s in self.sensors}
        self.reliability_age_decay = 0.98
        self.dwell_counters: Dict[str, int] = {s: 0 for s in self.sensors}
        self.total_switches: int = 0
        self.total_probes: int = 0

    def reset(self, initial_active: Optional[Dict[str, int]] = None) -> None:
        """Reset controller to initial conditions."""
        if initial_active is None:
            self.active_mask = {s: 1 if s == self.anchor_sensor else 0 for s in self.sensors}
        else:
            self.active_mask = dict(initial_active)

        self.last_observed_reliability = {s: 1.0 for s in self.sensors}
        self.windows_since_last_observed = {s: 0 for s in self.sensors}
        self.dwell_counters = {s: 0 for s in self.sensors}
        self.total_switches = 0
        self.total_probes = 0

    def compute_normalized_cost(self, active_mask: Optional[Dict[str, int]] = None) -> float:
        """
        Compute normalized sensing cost proxy:
            C(t) = sum_i [ a_i(t) * w_i ] / sum_i [ w_i ]
        """
        mask = active_mask if active_mask is not None else self.active_mask
        active_weight = sum(self.cost_weights[s] * mask[s] for s in self.sensors)
        return float(active_weight / max(1e-6, self.total_cost_weight))

    def step(
        self,
        current_reliabilities: Dict[str, float],
        fused_uncertainty: float
    ) -> Tuple[Dict[str, int], str, Dict[str, Any]]:
        """
        Evaluate window t conditions and determine active sensor configuration for window t+1.
        
        Returns:
            next_active_mask: active configuration for window t+1.
            action_name: KEEP, ACTIVATE, DEACTIVATE, PROBE.
            meta: telemetry and decision metrics.
        """
        # Update observation state for currently active sensors
        for s in self.sensors:
            if self.active_mask[s] == 1 and s in current_reliabilities:
                self.last_observed_reliability[s] = float(current_reliabilities[s])
                self.windows_since_last_observed[s] = 0
                self.dwell_counters[s] += 1
            else:
                self.windows_since_last_observed[s] += 1
                self.dwell_counters[s] += 1

        next_mask = dict(self.active_mask)
        action = "KEEP"
        reason = "Stable operation"

        active_sensors = [s for s in self.sensors if self.active_mask[s] == 1]
        inactive_sensors = [s for s in self.sensors if self.active_mask[s] == 0]

        # 1. Check if any active sensor is severely degraded
        unreliable_active = [
            s for s in active_sensors
            if self.last_observed_reliability[s] < self.r_low
        ]

        # 2. Check if prediction uncertainty is high
        high_uncertainty = fused_uncertainty >= self.u_high

        # 3. Check for periodic probe of inactive sensors
        probe_target = None
        for s in inactive_sensors:
            if self.windows_since_last_observed[s] >= self.probe_interval:
                probe_target = s
                break

        # --- Decision Logic ---
        if unreliable_active or high_uncertainty:
            # Need reinforcement: ACTIVATE best candidate inactive sensor
            affordable = [s for s in inactive_sensors if self.compute_normalized_cost({**next_mask, s: 1}) <= self.cost_budget]
            if affordable:
                # Prioritize inactive sensor with highest last-observed reliability
                candidate = max(affordable, key=lambda s:
                    self.last_observed_reliability[s] * self.reliability_age_decay ** self.windows_since_last_observed[s])
                next_mask[candidate] = 1
                self.dwell_counters[candidate] = 0
                self.total_switches += 1
                action = "ACTIVATE"
                reason = f"Activated {candidate} due to {'unreliable active ' + str(unreliable_active) if unreliable_active else 'high uncertainty'}"

            # If an active sensor is completely failed and another sensor is active, deactive failed sensor
            if len(active_sensors) > 1:
                for bad_s in unreliable_active:
                    if bad_s != self.anchor_sensor and self.dwell_counters[bad_s] >= self.min_dwell_windows:
                        next_mask[bad_s] = 0
                        self.dwell_counters[bad_s] = 0
                        self.total_switches += 1
                        action = "DEACTIVATE"
                        reason = f"Deactivated failing sensor {bad_s}"
                        break

        elif probe_target is not None and self.compute_normalized_cost({**next_mask, probe_target: 1}) <= self.cost_budget:
            # Probe inactive sensor to check for recovery
            next_mask[probe_target] = 1
            self.total_probes += 1
            self.total_switches += 1
            action = "PROBE"
            reason = f"Probing inactive sensor {probe_target} for health recovery"

        elif fused_uncertainty <= self.u_low and len(active_sensors) > 1:
            # System is confident and stable: consider DEACTIVATING redundant non-anchor sensor
            # Check dwell time
            deact_candidate = None
            for s in active_sensors:
                if s != self.anchor_sensor and self.dwell_counters[s] >= self.min_dwell_windows:
                    deact_candidate = s
                    break

            if deact_candidate is not None:
                next_mask[deact_candidate] = 0
                self.dwell_counters[deact_candidate] = 0
                self.total_switches += 1
                action = "DEACTIVATE"
                reason = f"Deactivated redundant {deact_candidate} to conserve normalized sensing cost"

        # Hard invariant: at least one sensor MUST be active
        next_mask[self.anchor_sensor] = 1

        self.active_mask = next_mask
        meta = {
            "action": action,
            "reason": reason,
            "next_active_mask": next_mask,
            "fused_uncertainty": float(fused_uncertainty),
            "normalized_cost": self.compute_normalized_cost(next_mask),
            "aged_inactive_reliability": {
                s: self.last_observed_reliability[s] * self.reliability_age_decay ** self.windows_since_last_observed[s]
                for s in self.sensors if not next_mask[s]
            },
            "total_switches": self.total_switches,
            "total_probes": self.total_probes
        }
        return next_mask, action, meta
