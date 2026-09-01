"""Speech recognition metrics."""

from callasr.metrics.wer import normalize_text, word_error_rate

__all__ = ["normalize_text", "word_error_rate"]
