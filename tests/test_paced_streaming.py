import importlib

import pytest


def _api():
    return importlib.import_module("callasr.paced_streaming")


def test_paced_streaming_public_api_exists() -> None:
    module = _api()

    assert module.PacedStreamingASRAdapter is not None
    assert module.PacedStreamingBenchmarkResult is not None
    assert callable(module.run_paced_streaming_benchmark)


@pytest.mark.parametrize("realtime_factor", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_realtime_factor_is_rejected(realtime_factor: float) -> None:
    module = _api()

    with pytest.raises(ValueError, match="realtime_factor"):
        module.validate_realtime_factor(realtime_factor)
