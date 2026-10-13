from typing import Any, Dict, List, Set
import networkx as nx


class DependencyGraphEngine:
    """NetworkX directed dependency graph modeling microservice hierarchies and failure cascades."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._initialize_default_topology()

    def _initialize_default_topology(self) -> None:
        edges = [
            ("edge-gateway", "auth-service"),
            ("edge-gateway", "catalog-service"),
            ("edge-gateway", "checkout-api"),
            ("checkout-api", "order-service"),
            ("order-service", "payment-service"),
            ("order-service", "inventory-service"),
            ("checkout-api", "payment-service"),
            ("checkout-api", "inventory-service"),
            ("checkout-api", "notification-service"),
        ]
        self.graph.add_edges_from(edges)

    def get_upstream_dependencies(self, service_id: str) -> List[str]:
        """Returns direct dependencies called by service_id."""
        if service_id not in self.graph:
            return []
        return list(self.graph.successors(service_id))

    def get_downstream_callers(self, service_id: str) -> List[str]:
        """Returns callers that depend on service_id."""
        if service_id not in self.graph:
            return []
        return list(self.graph.predecessors(service_id))

    def get_blast_radius(self, origin_service_id: str) -> List[str]:
        """Returns all services affected if origin_service_id degrades (reverse ancestors)."""
        if origin_service_id not in self.graph:
            return [origin_service_id]
        ancestors = nx.ancestors(self.graph, origin_service_id)
        return list(ancestors.union({origin_service_id}))

    def export_topology(self) -> Dict[str, Any]:
        nodes = [{"id": n, "name": n} for n in self.graph.nodes()]
        links = [{"source": u, "target": v} for u, v in self.graph.edges()]
        return {"nodes": nodes, "links": links}
