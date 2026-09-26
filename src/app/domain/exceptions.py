"""Domain-specific exceptions."""


class AudioRetrievalError(Exception):
    """Base class for errors raised by the audio retrieval system."""


class TranscriptionError(AudioRetrievalError):
    """Raised when an audio file cannot be transcribed."""


class DiarizationError(AudioRetrievalError):
    """Raised when speakers cannot be identified in an audio file."""


class ConfigurationError(AudioRetrievalError):
    """Raised when required configuration is missing or invalid."""


class GoldenDatasetError(AudioRetrievalError):
    """Raised when the golden query set or ground truth is inconsistent."""
