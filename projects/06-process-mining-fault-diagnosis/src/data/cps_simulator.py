"""
Cyber-Physical System (CPS) telemetry simulator with fault injection.

Simulates sensor/actuator time-series for:
  - Normal operation
  - Sensor drift
  - Actuator stuck
  - Communication delay
  - Oscillation faults
  - Overload conditions
  - Partial failures
  - Cascade failures

Also produces event logs in XES format compatible with pm4py process mining.
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CPS Configuration
# ─────────────────────────────────────────────────────────────────────────────

FAULT_TYPES = [
    "normal",
    "sensor_drift",
    "actuator_stuck",
    "communication_delay",
    "oscillation",
    "overload",
    "partial_failure",
    "cascade_failure",
]

FAULT_LABEL_MAP = {ft: i for i, ft in enumerate(FAULT_TYPES)}


@dataclass
class CPSConfig:
    num_sensors: int = 20
    num_actuators: int = 5
    sampling_rate_hz: float = 100.0
    window_size: int = 50
    noise_std: float = 0.02
    seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# CPS Telemetry Simulator
# ─────────────────────────────────────────────────────────────────────────────

class CPSFaultSimulator:
    """
    High-fidelity CPS fault simulator.
    Each fault type modifies the baseline sensor/actuator dynamics.
    """

    def __init__(self, config: CPSConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.dt = 1.0 / config.sampling_rate_hz

    # ── Baseline dynamics ────────────────────────────────────────────────────

    def _baseline_signal(self, t: np.ndarray, sensor_id: int) -> np.ndarray:
        """Generate baseline sensor signal: sum of harmonics + noise."""
        freq = 1.0 + sensor_id * 0.5
        phase = sensor_id * np.pi / self.config.num_sensors
        signal = (
            np.sin(2 * np.pi * freq * t + phase) * 0.5
            + np.sin(2 * np.pi * 3 * freq * t + phase * 1.5) * 0.1
            + self.rng.normal(0, self.config.noise_std, t.shape)
        )
        return signal.astype(np.float32)

    def _actuator_signal(self, t: np.ndarray, actuator_id: int) -> np.ndarray:
        """Generate piecewise-constant actuator command signal."""
        # Actuator switches every ~0.5 seconds
        period = 0.5 + actuator_id * 0.2
        level = np.floor(t / period) % 4 / 3.0  # 0, 1/3, 2/3, 1
        return (level + self.rng.normal(0, 0.01, t.shape)).astype(np.float32)

    # ── Fault injection functions ─────────────────────────────────────────────

    def _inject_sensor_drift(
        self, signal: np.ndarray, t: np.ndarray, drift_rate: float = 0.05
    ) -> np.ndarray:
        """Linear drift added to sensor channel."""
        return signal + (drift_rate * t).astype(np.float32)

    def _inject_actuator_stuck(
        self, signal: np.ndarray, t: np.ndarray, stuck_value: float = 0.5
    ) -> np.ndarray:
        """Actuator stuck at a fixed value after fault onset."""
        fault_onset = len(t) // 3
        result = signal.copy()
        result[fault_onset:] = stuck_value + self.rng.normal(0, 0.005, result[fault_onset:].shape)
        return result.astype(np.float32)

    def _inject_communication_delay(
        self, signal: np.ndarray, delay_steps: int = 5
    ) -> np.ndarray:
        """Simulate communication delay by shifting signal."""
        result = np.zeros_like(signal)
        result[delay_steps:] = signal[:-delay_steps]
        result[:delay_steps] = signal[0]
        return result.astype(np.float32)

    def _inject_oscillation(
        self, signal: np.ndarray, t: np.ndarray, osc_freq: float = 10.0
    ) -> np.ndarray:
        """High-frequency oscillation superimposed on normal signal."""
        osc = 0.3 * np.sin(2 * np.pi * osc_freq * t)
        return (signal + osc).astype(np.float32)

    def _inject_overload(
        self, signal: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """Signal clips at saturation level."""
        saturated = np.clip(signal * 2.5, -1.0, 1.0)
        fault_onset = len(t) // 4
        result = signal.copy()
        result[fault_onset:] = saturated[fault_onset:]
        return result.astype(np.float32)

    def _inject_partial_failure(
        self, signal: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """Intermittent signal dropout (partial sensor failure)."""
        result = signal.copy()
        dropout_mask = self.rng.random(len(t)) < 0.15  # 15% dropout
        result[dropout_mask] = 0.0
        return result.astype(np.float32)

    def _inject_cascade_failure(
        self, signals: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """
        Cascade failure: fault propagates from one sensor to neighboring sensors.
        signals: (num_sensors, time_steps)
        """
        result = signals.copy()
        cascade_start = len(t) // 5
        for i in range(self.config.num_sensors):
            onset = cascade_start + i * (len(t) // (self.config.num_sensors * 3))
            if onset < len(t):
                result[i, onset:] *= np.exp(-0.5 * (t[onset:] - t[onset]))
                result[i, onset:] += self.rng.normal(0, 0.2, result[i, onset:].shape)
        return result.astype(np.float32)

    # ── Main simulation method ────────────────────────────────────────────────

    def simulate_window(
        self,
        fault_type: str = "normal",
        window_size: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Simulate one window of CPS operation.

        Returns:
            sensor_data:    (window_size, num_sensors)
            actuator_data:  (window_size, num_actuators)
            metadata:       simulation metadata dict
        """
        W = window_size or self.config.window_size
        t = np.linspace(0, W * self.dt, W)

        # Generate baseline sensor signals
        sensor_signals = np.stack([
            self._baseline_signal(t, i) for i in range(self.config.num_sensors)
        ])  # (num_sensors, W)

        # Generate actuator signals
        actuator_signals = np.stack([
            self._actuator_signal(t, i) for i in range(self.config.num_actuators)
        ])  # (num_actuators, W)

        # Inject faults
        if fault_type == "sensor_drift":
            target = self.rng.integers(0, self.config.num_sensors)
            sensor_signals[target] = self._inject_sensor_drift(sensor_signals[target], t)

        elif fault_type == "actuator_stuck":
            target = self.rng.integers(0, self.config.num_actuators)
            actuator_signals[target] = self._inject_actuator_stuck(actuator_signals[target], t)

        elif fault_type == "communication_delay":
            for i in range(self.config.num_sensors // 4):
                delay = self.rng.integers(2, 10)
                sensor_signals[i] = self._inject_communication_delay(sensor_signals[i], delay)

        elif fault_type == "oscillation":
            for i in range(self.config.num_sensors // 2):
                osc_freq = self.rng.uniform(8.0, 15.0)
                sensor_signals[i] = self._inject_oscillation(sensor_signals[i], t, osc_freq)

        elif fault_type == "overload":
            for i in range(self.config.num_sensors):
                sensor_signals[i] = self._inject_overload(sensor_signals[i], t)

        elif fault_type == "partial_failure":
            targets = self.rng.integers(0, self.config.num_sensors, size=3)
            for tgt in targets:
                sensor_signals[tgt] = self._inject_partial_failure(sensor_signals[tgt], t)

        elif fault_type == "cascade_failure":
            sensor_signals = self._inject_cascade_failure(sensor_signals, t)

        # Transpose: (W, num_sensors) and (W, num_actuators)
        sensor_data   = sensor_signals.T.astype(np.float32)
        actuator_data = actuator_signals.T.astype(np.float32)

        metadata = {
            "fault_type": fault_type,
            "fault_label": FAULT_LABEL_MAP[fault_type],
            "window_size": W,
            "num_sensors": self.config.num_sensors,
            "num_actuators": self.config.num_actuators,
        }
        return sensor_data, actuator_data, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Event Log Generator (Process Mining)
# ─────────────────────────────────────────────────────────────────────────────

class CPSEventLogGenerator:
    """
    Converts CPS telemetry into event logs suitable for process mining.

    Event abstraction:
      - Each threshold crossing of a sensor = an event
      - Each actuator state change = an event
      - Each fault onset = a special event
    """

    def __init__(self, config: CPSConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def _abstract_sensor_events(
        self,
        sensor_data: np.ndarray,
        base_time: datetime.datetime,
        dt: float,
    ) -> list[dict]:
        """Convert sensor crossings to discrete events."""
        events = []
        W, N = sensor_data.shape
        thresholds = [0.3, 0.6, -0.3, -0.6]

        for sensor_id in range(N):
            signal = sensor_data[:, sensor_id]
            for threshold in thresholds:
                crossings = np.where(np.diff(np.sign(signal - threshold)))[0]
                for step in crossings:
                    direction = "rising" if signal[step + 1] > signal[step] else "falling"
                    events.append({
                        "case:concept:name": f"sensor_{sensor_id}",
                        "concept:name": f"S{sensor_id}_{direction}_{abs(int(threshold * 10))}",
                        "time:timestamp": base_time + datetime.timedelta(seconds=step * dt),
                        "sensor_id": sensor_id,
                        "threshold": threshold,
                        "value": float(signal[step]),
                    })
        return events

    def _abstract_actuator_events(
        self,
        actuator_data: np.ndarray,
        base_time: datetime.datetime,
        dt: float,
    ) -> list[dict]:
        """Convert actuator state changes to events."""
        events = []
        W, M = actuator_data.shape
        for act_id in range(M):
            signal = actuator_data[:, act_id]
            quantized = np.round(signal * 4) / 4
            changes = np.where(np.abs(np.diff(quantized)) > 0.2)[0]
            for step in changes:
                events.append({
                    "case:concept:name": f"actuator_{act_id}",
                    "concept:name": f"A{act_id}_change_{int(quantized[step + 1] * 100)}",
                    "time:timestamp": base_time + datetime.timedelta(seconds=step * dt),
                    "actuator_id": act_id,
                    "from_level": float(quantized[step]),
                    "to_level": float(quantized[step + 1]),
                })
        return events

    def generate_event_log(
        self,
        sensor_data: np.ndarray,
        actuator_data: np.ndarray,
        fault_type: str,
        case_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate pm4py-compatible event log DataFrame from CPS telemetry.
        """
        dt = 1.0 / self.config.sampling_rate_hz
        base_time = datetime.datetime(2024, 1, 1, 0, 0, 0)
        case_id = case_id or str(uuid.uuid4())[:8]

        events = (
            self._abstract_sensor_events(sensor_data, base_time, dt)
            + self._abstract_actuator_events(actuator_data, base_time, dt)
        )

        if not events:
            # Ensure at least one event
            events.append({
                "case:concept:name": "system",
                "concept:name": "normal_tick",
                "time:timestamp": base_time,
            })

        df = pd.DataFrame(events)
        df["case:concept:name"] = f"case_{case_id}_" + df["case:concept:name"]
        df["fault_type"] = fault_type
        df["fault_label"] = FAULT_LABEL_MAP[fault_type]
        df = df.sort_values("time:timestamp").reset_index(drop=True)
        return df

    def extract_process_features(self, event_log: pd.DataFrame) -> np.ndarray:
        """
        Extract statistical process features from event log for ML:
          - Event frequency per type
          - Inter-arrival time statistics
          - Trace complexity metrics
          - Variant count
        """
        features = []

        # Event frequency (top 20 event types)
        event_counts = event_log["concept:name"].value_counts()
        top_events = event_counts.head(20)
        freq_features = np.zeros(20, dtype=np.float32)
        for i, count in enumerate(top_events.values[:20]):
            freq_features[i] = float(count)

        # Inter-arrival times
        if len(event_log) > 1:
            timestamps = pd.to_datetime(event_log["time:timestamp"])
            inter_arrivals = timestamps.diff().dt.total_seconds().dropna().values
            iat_features = np.array([
                float(inter_arrivals.mean()),
                float(inter_arrivals.std()),
                float(np.percentile(inter_arrivals, 25)),
                float(np.percentile(inter_arrivals, 75)),
                float(inter_arrivals.min()),
                float(inter_arrivals.max()),
            ], dtype=np.float32)
        else:
            iat_features = np.zeros(6, dtype=np.float32)

        # Trace complexity
        num_unique_events = float(event_log["concept:name"].nunique())
        num_events = float(len(event_log))
        num_cases = float(event_log["case:concept:name"].nunique())
        complexity_features = np.array([
            num_unique_events, num_events, num_cases,
            num_events / max(num_cases, 1),
        ], dtype=np.float32)

        features = np.concatenate([freq_features, iat_features, complexity_features])
        return features.astype(np.float32)
