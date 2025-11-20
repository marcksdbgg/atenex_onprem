"""Deprecated shim kept for backwards compatibility.

Historically the sparse search service used Google Cloud Storage to persist the
BM25 indexes. After migrating to MinIO, the codebase now relies on the
``MinioIndexStorageAdapter`` implementation. This module simply re-exports the
new adapter using the old class names so that any remaining imports keep
working. It can be removed once all references are updated.
"""

from app.infrastructure.storage.minio_index_storage_adapter import (  # noqa: F401
    MinioIndexStorageAdapter as GCSIndexStorageAdapter,
    MinioIndexStorageError as GCSIndexStorageError,
)

__all__ = ["GCSIndexStorageAdapter", "GCSIndexStorageError"]