"""Canonical Dual Raw scorer used by the current project method."""

from __future__ import annotations

from .evaluator import CCRConfig, CompressionEvaluator
from .aggregation import MinimumScoreEvaluator


class DualRawEvaluator:
    """Compute the minimum raw conditional compression score.

    The configuration is intentionally frozen here so the project default cannot
    silently drift back to the historical V67 overhead-corrected score.
    """

    method_name = "Dual Raw"

    def __init__(self) -> None:
        shared = {"lowercase": False, "strip": True, "separator": "\n"}
        self.lz4 = CompressionEvaluator(
            CCRConfig(compressor="lz4", lz4_level=16, **shared)
        )
        self.zstd = CompressionEvaluator(
            CCRConfig(compressor="zstd", zstd_level=19, **shared)
        )
        self._minimum = MinimumScoreEvaluator((self.lz4, self.zstd))

    @property
    def config(self) -> dict[str, object]:
        """Return the immutable method configuration for audit output."""

        return {
            "method": "dual_raw",
            "compressors": {
                "lz4": {"mode": "frame", "level": 16},
                "zstd": {"level": 19},
            },
            "encoding": "utf-8",
            "lowercase": False,
            "strip": True,
            "separator": "newline",
            "aggregation": "minimum",
            "overhead_correction": False,
        }

    def evaluate(self, context: str, summary: str) -> dict:
        result = self._minimum.evaluate(context, summary)
        return {
            **result,
            "method": self.method_name,
            "formula": "min(H_raw_LZ4, H_raw_Zstd)",
            "compressors": ["LZ4", "Zstd"],
            "overhead_correction": False,
        }
