"""Minimal score aggregation used by CompFaith."""

from __future__ import annotations


class MinimumScoreEvaluator:
    """Return the minimum score across the fixed compressor views."""

    def __init__(self, evaluators: tuple[object, ...]) -> None:
        if not evaluators:
            raise ValueError("evaluators must be non-empty")
        self.evaluators = evaluators

    def evaluate(self, context: str, summary: str) -> dict:
        component_results = [
            evaluator.evaluate(context, summary) for evaluator in self.evaluators
        ]
        component_scores = [float(result["score"]) for result in component_results]
        return {
            "score": min(component_scores),
            "component_scores": component_scores,
            "aggregation": "minimum",
        }
