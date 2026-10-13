import pytest
from services.incident_engine.graph import DependencyGraphEngine

def test_dependency_graph_traversal():
    graph = DependencyGraphEngine()
    
    # checkout-api calls payment-service
    upstreams = graph.get_upstream_dependencies("checkout-api")
    assert "payment-service" in upstreams
    assert "inventory-service" in upstreams

    # edge-gateway calls checkout-api
    callers = graph.get_downstream_callers("checkout-api")
    assert "edge-gateway" in callers

    # blast radius of payment-service includes callers checkout-api and edge-gateway
    blast = graph.get_blast_radius("payment-service")
    assert "payment-service" in blast
    assert "checkout-api" in blast
    assert "edge-gateway" in blast
