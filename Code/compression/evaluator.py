from __future__ import annotations

import bz2
import gzip
import lzma
from dataclasses import dataclass

from .normalization import normalize_text

try:
    import zstandard as zstd
except ModuleNotFoundError:  # pragma: no cover - only used without the declared dependency
    zstd = None

try:
    import brotli
except ModuleNotFoundError:  # pragma: no cover - only used without the declared dependency
    brotli = None

try:
    import lz4.frame as lz4_frame
except ModuleNotFoundError:  # pragma: no cover - only used without the declared dependency
    lz4_frame = None

try:
    import lz4.block as lz4_block
except ModuleNotFoundError:  # pragma: no cover - only used without the declared dependency
    lz4_block = None


@dataclass(frozen=True)
class CCRConfig:
    compressor: str = "lzma"
    lzma_preset: int = 9
    gzip_level: int = 9
    bz2_level: int = 9
    zstd_level: int = 19
    zstd_window_log: int | None = None
    brotli_quality: int = 11
    brotli_lgwin: int = 22
    brotli_mode: str = "text"
    lz4_level: int = 16
    lz4_block_size: str = "default"
    lz4_mode: str = "frame"
    lz4_store_size: bool = False
    lz4_block_checksum: bool = False
    lowercase: bool = True
    strip: bool = True
    separator: str = " "
    clip_min: float | None = 0.0
    clip_max: float | None = 1.0


class CompressionEvaluator:
    """Nonparametric hallucination evaluator based on Conditional Compression Ratio (CCR).

    The CCR score is defined as the ratio of compression-estimated conditional
    information of the generated text given the context to its self-information:

        Δ_bounded = min(C(y), max(0, C(x+y) - C(x)))
        CCR = Δ_bounded / C(y)

    where C(·) is the compressed byte length, x is the context, and y is the summary.

    A score close to 0 indicates the summary is mostly covered by the context (faithful),
    while a score close to 1 indicates the summary contains largely new information (hallucinated).
    """

    def __init__(self, config: CCRConfig | None = None):
        self.config = config or CCRConfig()

    def _compress_len(self, text: str) -> int:
        raw = text.encode("utf-8")
        compressor = self.config.compressor.lower()

        if compressor == "lzma":
            return len(lzma.compress(raw, preset=self.config.lzma_preset))
        if compressor == "gzip":
            return len(gzip.compress(raw, compresslevel=self.config.gzip_level))
        if compressor == "bz2":
            return len(bz2.compress(raw, compresslevel=self.config.bz2_level))
        if compressor == "zstd":
            if zstd is None:
                raise RuntimeError("zstandard is required for compressor='zstd'")
            if self.config.zstd_window_log is None:
                compressor_instance = zstd.ZstdCompressor(level=self.config.zstd_level)
            else:
                parameters = zstd.ZstdCompressionParameters.from_level(
                    self.config.zstd_level,
                    window_log=self.config.zstd_window_log,
                )
                compressor_instance = zstd.ZstdCompressor(compression_params=parameters)
            return len(compressor_instance.compress(raw))
        if compressor == "brotli":
            if brotli is None:
                raise RuntimeError("Brotli is required for compressor='brotli'")
            mode = {
                "generic": brotli.MODE_GENERIC,
                "text": brotli.MODE_TEXT,
                "font": brotli.MODE_FONT,
            }.get(self.config.brotli_mode.lower())
            if mode is None:
                raise ValueError(f"Unsupported Brotli mode: {self.config.brotli_mode}")
            return len(
                brotli.compress(
                    raw,
                    mode=mode,
                    quality=self.config.brotli_quality,
                    lgwin=self.config.brotli_lgwin,
                )
            )
        if compressor == "lz4":
            if self.config.lz4_mode.lower() == "block":
                if lz4_block is None:
                    raise RuntimeError("lz4 is required for compressor='lz4'")
                return len(
                    lz4_block.compress(
                        raw,
                        mode="high_compression",
                        compression=self.config.lz4_level,
                        store_size=False,
                    )
                )
            if lz4_frame is None:
                raise RuntimeError("lz4 is required for compressor='lz4'")
            block_size = {
                "default": lz4_frame.BLOCKSIZE_DEFAULT,
                "64kb": lz4_frame.BLOCKSIZE_MAX64KB,
                "256kb": lz4_frame.BLOCKSIZE_MAX256KB,
                "1mb": lz4_frame.BLOCKSIZE_MAX1MB,
                "4mb": lz4_frame.BLOCKSIZE_MAX4MB,
            }.get(self.config.lz4_block_size.lower())
            if block_size is None:
                raise ValueError(f"Unsupported LZ4 block size: {self.config.lz4_block_size}")
            return len(
                lz4_frame.compress(
                    raw,
                    compression_level=self.config.lz4_level,
                    block_size=block_size,
                    block_linked=True,
                    store_size=self.config.lz4_store_size,
                    block_checksum=self.config.lz4_block_checksum,
                    content_checksum=False,
                )
            )

        raise ValueError(f"Unsupported compressor: {self.config.compressor}")

    def evaluate(self, context: str, summary: str) -> dict[str, float]:
        context_norm = normalize_text(
            context,
            lowercase=self.config.lowercase,
            strip=self.config.strip,
        )
        summary_norm = normalize_text(
            summary,
            lowercase=self.config.lowercase,
            strip=self.config.strip,
        )

        if not summary_norm:
            len_c = float(self._compress_len(context_norm))
            return {
                "score": 0.0,
                "delta": 0.0,
                "len_context": len_c,
                "len_summary": 0.0,
                "len_combined": len_c,
                "delta_raw": 0.0,
                "delta_positive": 0.0,
                "delta_bounded": 0.0,
                "summary_information": 0.0,
                "conditional_information": 0.0,
                "raw_ratio": 0.0,
                "positive_ratio": 0.0,
            }

        len_context = self._compress_len(context_norm)
        len_summary = self._compress_len(summary_norm)
        combined = context_norm + self.config.separator + summary_norm
        len_combined = self._compress_len(combined)

        # Raw conditional compression increment
        delta_raw = len_combined - len_context

        # Remove negative compression artifacts
        delta_positive = max(0, delta_raw)

        # Bound conditional information by summary's self-information
        delta_bounded = min(len_summary, delta_positive)

        # Information quantities
        summary_information = len_summary
        conditional_information = delta_bounded

        # Final CCR score: ratio of conditional information to summary information
        score = conditional_information / summary_information if summary_information > 0 else 0.0

        # Engineering-safety clip (theoretically unnecessary due to delta_bounded)
        if self.config.clip_min is not None:
            score = max(self.config.clip_min, score)
        if self.config.clip_max is not None:
            score = min(self.config.clip_max, score)

        return {
            "score": float(score),
            "delta": float(delta_bounded),  # backward-compatible alias
            "len_context": float(len_context),
            "len_summary": float(len_summary),
            "len_combined": float(len_combined),
            "delta_raw": float(delta_raw),
            "delta_positive": float(delta_positive),
            "delta_bounded": float(delta_bounded),
            "summary_information": float(summary_information),
            "conditional_information": float(conditional_information),
            "raw_ratio": float(delta_raw / len_summary) if len_summary > 0 else 0.0,
            "positive_ratio": float(delta_positive / len_summary) if len_summary > 0 else 0.0,
        }
