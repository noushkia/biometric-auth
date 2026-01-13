"""Iris Encoder - Biometric iris encoding for authentication."""

__version__ = "0.1.0"


def __getattr__(name):
    """Lazy import to avoid loading cv2 until needed."""
    if name == "IrisPipeline":
        from .pipeline import IrisPipeline
        return IrisPipeline
    elif name == "EncodingResult":
        from .pipeline import EncodingResult
        return EncodingResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["IrisPipeline", "EncodingResult", "__version__"]
