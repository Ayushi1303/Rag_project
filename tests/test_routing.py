# =============================================================
#  tests/test_routing.py — Unit Tests for Routing Logic
#
#  Run with:  pytest tests/ -v
#
#  Tests the route_decision + router_node without making any
#  API calls. All tests are offline and free.
# =============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from nodes import router_node, route_decision


# ── Helpers ───────────────────────────────────────────────────

def make_state(
    answer: str = "Here is the answer.",
    query: str = "what is the return policy",
    chunks: list = None,
    error: str = None,
    confidence: str = "high",
) -> dict:
    """Build a minimal GraphState dict for testing."""
    return {
        "query": query,
        "clean_query": query,
        "answer": answer,
        "context": "some context",
        "context_chunks": chunks if chunks is not None else ["chunk1", "chunk2"],
        "sources": [{"source": "doc.pdf", "page": 1}],
        "confidence": confidence,
        "route": "",
        "human_answer": None,
        "error": error,
    }


# ── Tests: router_node sets state["route"] correctly ──────────

class TestRouterNode:

    def test_confident_answer_routes_to_answer(self):
        state = make_state(answer="You can return items within 30 days of purchase.")
        result = router_node(state)
        assert result["route"] == "answer"

    def test_empty_chunks_routes_to_escalate(self):
        state = make_state(chunks=[])
        result = router_node(state)
        assert result["route"] == "escalate"

    def test_short_answer_routes_to_escalate(self):
        state = make_state(answer="No.")   # < MIN_ANSWER_LENGTH
        result = router_node(state)
        assert result["route"] == "escalate"

    def test_uncertainty_phrase_routes_to_escalate(self):
        state = make_state(
            answer="I don't have enough information to answer this question."
        )
        result = router_node(state)
        assert result["route"] == "escalate"

    def test_uncertainty_phrase_variant(self):
        state = make_state(answer="I'm not sure about the specific details here.")
        result = router_node(state)
        assert result["route"] == "escalate"

    def test_not_mentioned_phrase_routes_to_escalate(self):
        state = make_state(answer="This topic is not mentioned in the context provided.")
        result = router_node(state)
        assert result["route"] == "escalate"

    def test_sensitive_keyword_refund_escalates(self):
        state = make_state(
            query="I want a refund for my broken product",
            answer="Our return policy allows returns within 30 days of purchase.",
        )
        result = router_node(state)
        assert result["route"] == "escalate"

    def test_sensitive_keyword_lawsuit_escalates(self):
        state = make_state(
            query="Can I file a lawsuit against the company?",
            answer="Please contact our support team for more information.",
        )
        result = router_node(state)
        assert result["route"] == "escalate"

    def test_sensitive_keyword_legal_escalates(self):
        state = make_state(
            query="What are my legal rights regarding this product?",
            answer="Here is some relevant policy information about product rights.",
        )
        result = router_node(state)
        assert result["route"] == "escalate"

    def test_upstream_error_escalates(self):
        state = make_state(error="API call failed")
        result = router_node(state)
        assert result["route"] == "escalate"

    def test_normal_answer_does_not_escalate(self):
        state = make_state(
            query="how do I track my order",
            answer="You can track your order by logging into your account and visiting the Orders section.",
        )
        result = router_node(state)
        assert result["route"] == "answer"


# ── Tests: route_decision reads state["route"] correctly ──────

class TestRouteDecision:

    def test_returns_answer_when_route_is_answer(self):
        state = make_state()
        state["route"] = "answer"
        assert route_decision(state) == "answer"

    def test_returns_escalate_when_route_is_escalate(self):
        state = make_state()
        state["route"] = "escalate"
        assert route_decision(state) == "escalate"

    def test_defaults_to_escalate_on_missing_route(self):
        state = make_state()
        state["route"] = ""
        # Empty string is falsy; route_decision defaults to "escalate"
        assert route_decision(state) == "escalate"


# ── Tests: edge cases ─────────────────────────────────────────

class TestEdgeCases:

    def test_case_insensitive_uncertainty_detection(self):
        # The LLM might capitalise phrases
        state = make_state(answer="I DON'T HAVE ENOUGH INFORMATION TO ANSWER THIS.")
        result = router_node(state)
        assert result["route"] == "escalate"

    def test_long_confident_answer_passes(self):
        long_answer = (
            "Our standard return policy allows customers to return any unused "
            "product in its original packaging within 30 calendar days of the "
            "purchase date. To initiate a return, please contact our support "
            "team with your order number and reason for return."
        )
        state = make_state(answer=long_answer, query="what is the return policy")
        result = router_node(state)
        assert result["route"] == "answer"

    def test_sensitive_keyword_in_confident_answer_still_escalates(self):
        # Even if the LLM gives a good-sounding answer, sensitive queries escalate
        state = make_state(
            query="I want to file a complaint about my order",
            answer=(
                "We take all complaints very seriously. Please submit your "
                "complaint through our official complaints portal with your "
                "order number and a description of the issue."
            ),
        )
        result = router_node(state)
        assert result["route"] == "escalate"
