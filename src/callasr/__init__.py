"""Public API for call-asr-bench."""

from callasr.adapters.vllm_realtime import VLLMRealtimeAdapter, VLLMRealtimeError
from callasr.audio import (
    AudioBuffer,
    apply_additive_noise,
    apply_gain_and_clip,
    resample,
    telephone_channel,
)
from callasr.codecs.g711 import decode_g711, encode_g711
from callasr.concurrent import (
    ConcurrentBenchmarkError,
    ConcurrentBenchmarkResult,
    ConcurrentItemResult,
    run_concurrent_benchmark,
)
from callasr.impairments import apply_jitter_loss, apply_packet_loss
from callasr.metrics.entities import (
    NumericEntity,
    NumericEntityScore,
    extract_numeric_entities,
    score_numeric_entities,
)
from callasr.metrics.wer import (
    ErrorCounts,
    character_error_counts,
    character_error_rate,
    micro_average,
    normalize_text,
    word_error_counts,
    word_error_rate,
)
from callasr.streaming import (
    ObservedStreamingUpdate,
    StreamingASRAdapter,
    StreamingBenchmarkResult,
    StreamingError,
    StreamingUpdate,
    frame_audio,
    partial_stability_score,
    run_streaming_benchmark,
)
from callasr.streaming_dataset import (
    StreamingDatasetError,
    StreamingDatasetItemResult,
    StreamingDatasetResult,
    StreamingDatasetSummary,
    run_streaming_dataset_benchmark,
)

__all__ = [
    "AudioBuffer",
    "ConcurrentBenchmarkError",
    "ConcurrentBenchmarkResult",
    "ConcurrentItemResult",
    "ErrorCounts",
    "NumericEntity",
    "NumericEntityScore",
    "ObservedStreamingUpdate",
    "StreamingASRAdapter",
    "StreamingBenchmarkResult",
    "StreamingDatasetError",
    "StreamingDatasetItemResult",
    "StreamingDatasetResult",
    "StreamingDatasetSummary",
    "StreamingError",
    "StreamingUpdate",
    "VLLMRealtimeAdapter",
    "VLLMRealtimeError",
    "apply_additive_noise",
    "apply_gain_and_clip",
    "apply_jitter_loss",
    "apply_packet_loss",
    "character_error_counts",
    "character_error_rate",
    "decode_g711",
    "encode_g711",
    "extract_numeric_entities",
    "frame_audio",
    "micro_average",
    "normalize_text",
    "partial_stability_score",
    "resample",
    "run_concurrent_benchmark",
    "run_streaming_benchmark",
    "run_streaming_dataset_benchmark",
    "score_numeric_entities",
    "telephone_channel",
    "word_error_counts",
    "word_error_rate",
]
