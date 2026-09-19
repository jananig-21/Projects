"""
Process Mining module using pm4py for:
1. Process model discovery (Alpha/Heuristics/Inductive Miner)
2. Conformance checking (token replay)
3. Directly-Follows Graph construction
4. Fault-specific process variant analysis
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

try:
    import pm4py
    from pm4py.objects.log.obj import EventLog
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.algo.discovery.alpha import algorithm as alpha_miner
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    from pm4py.statistics.directly_follows.log import get as dfg_discovery
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    log.warning("pm4py not available — using fallback process statistics")


class ProcessModelDiscovery:
    """
    Discovers Petri net process models from CPS event logs.
    Supports Alpha, Heuristics, and Inductive Miner algorithms.
    """

    ALGORITHMS = {"alpha", "heuristics", "inductive"}

    def __init__(self, algorithm: str = "heuristics") -> None:
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {self.ALGORITHMS}")
        self.algorithm   = algorithm
        self.petri_net   = None
        self.initial_marking = None
        self.final_marking   = None

    def _df_to_event_log(self, df: pd.DataFrame) -> "EventLog":
        """Convert pandas DataFrame to pm4py EventLog object."""
        df = df.rename(columns={
            "case_id":   "case:concept:name",
            "activity":  "concept:name",
            "timestamp": "time:timestamp",
        })
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
        return log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG)

    def discover(self, event_log_df: pd.DataFrame) -> dict:
        """
        Discover a process model from the event log.
        Returns discovered model metadata.
        """
        if not PM4PY_AVAILABLE:
            return self._fallback_discover(event_log_df)

        log_obj = self._df_to_event_log(event_log_df)

        if self.algorithm == "alpha":
            net, im, fm = alpha_miner.apply(log_obj)
        elif self.algorithm == "heuristics":
            net, im, fm = heuristics_miner.apply(log_obj)
        elif self.algorithm == "inductive":
            net, im, fm = inductive_miner.apply(log_obj)

        self.petri_net       = net
        self.initial_marking = im
        self.final_marking   = fm

        return {
            "algorithm":   self.algorithm,
            "num_places":  len(net.places),
            "num_transitions": len(net.transitions),
            "num_arcs":    len(net.arcs),
        }

    def _fallback_discover(self, df: pd.DataFrame) -> dict:
        """Fallback when pm4py unavailable: compute DFG statistics."""
        activities     = df["activity"].value_counts().to_dict()
        transitions    = df.groupby("case_id")["activity"].apply(
            lambda g: list(zip(g.iloc[:-1], g.iloc[1:]))
        ).explode().value_counts().to_dict() if len(df) > 0 else {}
        return {
            "algorithm":       "fallback_dfg",
            "num_activities":  len(activities),
            "num_transitions": len(transitions),
        }

    def check_conformance(self, test_log_df: pd.DataFrame) -> dict:
        """Token replay conformance checking."""
        if not PM4PY_AVAILABLE or self.petri_net is None:
            return self._fallback_conformance(test_log_df)

        log_obj = self._df_to_event_log(test_log_df)
        replayed = token_replay.apply(
            log_obj, self.petri_net, self.initial_marking, self.final_marking
        )
        fitnesses = [t["trace_fitness"] for t in replayed if "trace_fitness" in t]
        return {
            "mean_fitness":   float(np.mean(fitnesses)) if fitnesses else 0.0,
            "min_fitness":    float(np.min(fitnesses)) if fitnesses else 0.0,
            "max_fitness":    float(np.max(fitnesses)) if fitnesses else 0.0,
            "num_traces":     len(replayed),
            "perfectly_fitting": sum(1 for f in fitnesses if f >= 0.99),
        }

    def _fallback_conformance(self, df: pd.DataFrame) -> dict:
        return {"mean_fitness": 0.0, "note": "pm4py unavailable"}


class DirectlyFollowsGraphAnalyzer:
    """
    Builds and analyzes Directly-Follows Graphs (DFG) for fault pattern detection.
    DFG shows how activities follow one another in the event log.
    """

    def build_dfg(
        self, event_log_df: pd.DataFrame
    ) -> dict[tuple[str, str], int]:
        """Build DFG from event log DataFrame."""
        dfg: dict[tuple[str, str], int] = {}
        for case_id, group in event_log_df.groupby("case_id"):
            activities = group.sort_values("timestamp")["activity"].tolist()
            for a, b in zip(activities[:-1], activities[1:]):
                key       = (a, b)
                dfg[key]  = dfg.get(key, 0) + 1
        return dfg

    def dfg_to_feature_vector(
        self,
        dfg: dict[tuple[str, str], int],
        activity_vocab: Optional[list[str]] = None,
    ) -> np.ndarray:
        """
        Convert DFG edge frequencies to a fixed-size feature vector.
        """
        if activity_vocab is None:
            activity_vocab = [
                "STABLE", "MODERATE_ACTIVITY", "OSCILLATION",
                "HIGH_VIBRATION", "CRITICAL_SPIKE",
            ]
        n     = len(activity_vocab)
        idx   = {a: i for i, a in enumerate(activity_vocab)}
        mat   = np.zeros((n, n), dtype=np.float32)

        for (src, dst), cnt in dfg.items():
            if src in idx and dst in idx:
                mat[idx[src], idx[dst]] = cnt

        # Normalize
        row_sums = mat.sum(axis=1, keepdims=True) + 1e-8
        return (mat / row_sums).flatten()

    def compute_fault_dfg_signatures(
        self,
        event_log_df: pd.DataFrame,
    ) -> dict[int, np.ndarray]:
        """
        Compute DFG feature vectors per fault class.
        Returns {fault_label: feature_vector}.
        """
        signatures = {}
        for label, group in event_log_df.groupby("fault_label"):
            dfg = self.build_dfg(group)
            signatures[int(label)] = self.dfg_to_feature_vector(dfg)
        return signatures
