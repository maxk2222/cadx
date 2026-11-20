"""Typed domain errors for the project media upload path.

The route layer maps these to exact HTTP status codes (never by string
inspection):

    UnsupportedMediaTypeError -> 415
    PayloadTooLargeError      -> 413
    InvalidUploadError        -> 400
    GuideNotFoundError        -> 404
    ConverterUnavailableError -> 503
    StorageRecoveryError      -> 500 (logged; backup preserved on disk)
"""

from __future__ import annotations


class UploadError(Exception):
    """Base class for all media-upload domain errors."""


class InvalidUploadError(UploadError):
    """Invalid id, filename, or file content."""


class UnsupportedMediaTypeError(UploadError):
    """File extension is not in the allowlist."""


class PayloadTooLargeError(UploadError):
    """Streamed upload exceeded the configured size cap."""


class GuideNotFoundError(UploadError):
    """The target guide does not exist."""


class ConverterUnavailableError(UploadError):
    """Docker or the Assimp image can't be used, so a non-GLB model can't be converted right now."""


class StorageRecoveryError(UploadError):
    """Catastrophic: restoring the previous media from its backup failed.

    The backup file is deliberately PRESERVED on disk for manual recovery —
    safety over cosmetic cleanup.
    """
