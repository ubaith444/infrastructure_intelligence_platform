from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from services.incident_engine.graph import DependencyGraphEngine
from storage.models import AnomalyRecord


class CandidateOriginRanker:
    """Scores candidate failure origins across directed dependency graphs using 5 associative factors:
    1. Temporal precedence (earliest anomaly onset)
    2. Topological depth (upstream dependency of failing callers)
    3. Shared traces & error propagation patterns (e.g., checkout latency increased after payment errors rose)
    4. Recent deployment or configuration change proximity
    5. Anomaly severity and temporal persistence
    
    Linguistic Governance: Strictly maintains associative phrasing without claiming unverified causation.
    """

    def __init__(self, graph_engine: DependencyGraphEngine):
        self.graph = graph_engine
        self.recent_deployments: Dict[str, float] = {}

    def record_deployment(self, service_id: str, timestamp: float) -> None:
        self.recent_deployments[service_id] = timestamp

    def rank_candidates(
        self,
        anomalies: List[AnomalyRecord],
        damping_factors: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        if not anomalies:
            return []

        damping = damping_factors or {}
        # Group anomalies by service
        svc_anoms: Dict[str, List[AnomalyRecord]] = defaultdict(list)
        for a in anomalies:
            svc_anoms[a.service_id].append(a)

        candidates = []
        earliest_time = min(a.timestamp.timestamp() for a in anomalies)

        for svc, a_list in svc_anoms.items():
            first_seen = min(a.timestamp.timestamp() for a in a_list)
            persistence_count = len(a_list)

            # Factor 1: Temporal precedence
            time_score = 1.0 / (1.0 + max(0.0, first_seen - earliest_time) / 30.0)

            # Factor 2: Topological depth (upstream dependency of other failing services)
            downstream = self.graph.get_downstream_callers(svc)
            impacted_downstream = [d for d in downstream if d in svc_anoms]
            topo_score = min(1.0, 0.4 + 0.3 * len(impacted_downstream))

            # Factor 3: Anomaly severity & persistence
            max_sev = max(a.score for a in a_list)
            sev_score = min(1.0, (max_sev / 5.0) * (0.8 + 0.2 * min(3, persistence_count) / 3.0))

            # Factor 4: Recent deployment proximity
            dep_score = 0.85 if svc in self.recent_deployments else 0.10

            # Factor 5: Error propagation & shared trace evidence
            propagation_evidence: List[str] = []
            error_anoms = [a for a in a_list if "err" in a.narrative.lower() or a.metric_attributions.get("error_rate", 0) > 0.3]
            has_error_spike = len(error_anoms) > 0

            for caller in impacted_downstream:
                caller_anoms = svc_anoms[caller]
                caller_first = min(ca.timestamp.timestamp() for ca in caller_anoms)
                caller_lat_anoms = [ca for ca in caller_anoms if "latency" in ca.narrative.lower() or ca.metric_attributions.get("latency_p90", 0) > 0.3]
                
                if caller_first >= first_seen:
                    if has_error_spike and caller_lat_anoms:
                        propagation_evidence.append(
                            f"{caller} latency increased after {svc} errors rose"
                        )
                    else:
                        propagation_evidence.append(
                            f"{caller} degradation followed {svc} anomaly onset"
                        )

            propagation_score = min(1.0, 0.5 + 0.25 * len(propagation_evidence))

            # Combined associative confidence weighting
            # Weights: Temporal (30%), Topology (25%), Propagation Evidence (20%), Severity (15%), Deployment (10%)
            raw_confidence = (
                (0.30 * time_score)
                + (0.25 * topo_score)
                + (0.20 * propagation_score)
                + (0.15 * sev_score)
                + (0.10 * dep_score)
            )
            damped_confidence = min(0.99, raw_confidence * damping.get(svc, 1.0))

            # Associative explanation narrative adhering to linguistic governance
            evidence_summary = (
                f" Observed propagation: {'; '.join(propagation_evidence)}."
                if propagation_evidence
                else ""
            )
            callers_str = f"Upstream dependency of {', '.join(downstream)}" if downstream else "Edge entry point"
            time_str = datetime.fromtimestamp(first_seen).strftime("%H:%M:%S")

            narrative = (
                f"Service '{svc}' is strongly associated with the observed failure cluster "
                f"(candidate origin confidence: {damped_confidence * 100:.1f}%). "
                f"Earliest anomaly recorded at {time_str}. {callers_str}.{evidence_summary} "
                f"(Associative evidence trail; observational telemetry does not assert unverified causality)."
            )

            candidates.append({
                "service_id": svc,
                "confidence": round(damped_confidence, 4),
                "temporal_score": round(time_score, 3),
                "topology_score": round(topo_score, 3),
                "propagation_score": round(propagation_score, 3),
                "severity_score": round(sev_score, 3),
                "persistence_count": persistence_count,
                "propagation_evidence": propagation_evidence,
                "associative_narrative": narrative,
                "blast_radius": self.graph.get_blast_radius(svc),
            })

        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        return candidates
