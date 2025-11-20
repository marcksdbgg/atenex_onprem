# sparse-search-service/app/infrastructure/storage/minio_index_storage_adapter.py
"""Adapter that stores BM25 indexes inside a MinIO/S3 compatible bucket."""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Tuple, Optional, Any

import structlog
from minio import Minio
from minio.error import S3Error
from pydantic import SecretStr

from app.application.ports.sparse_index_storage_port import SparseIndexStoragePort
from app.core.config import settings

log = structlog.get_logger(__name__)

BM25_DUMP_FILENAME = "bm25_index.bm2s"
ID_MAP_FILENAME = "id_map.json"
INDEX_ROOT_PATH = "indices"


class MinioIndexStorageError(Exception):
    """Raised when the object storage interaction fails."""


class MinioIndexStorageAdapter(SparseIndexStoragePort):
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[Any] = None,
        secure: Optional[bool] = None,
        region: Optional[str] = None,
    ) -> None:
        self.bucket_name = bucket_name or settings.INDEX_STORAGE_BUCKET_NAME
        endpoint_to_use = endpoint or settings.INDEX_STORAGE_ENDPOINT
        access_key_to_use = access_key or settings.INDEX_STORAGE_ACCESS_KEY
        secret_key_to_use = secret_key or settings.INDEX_STORAGE_SECRET_KEY
        secure_flag = settings.INDEX_STORAGE_SECURE if secure is None else secure
        region_to_use = region or settings.INDEX_STORAGE_REGION

        if isinstance(secret_key_to_use, SecretStr):
            secret_key_to_use = secret_key_to_use.get_secret_value()
        if isinstance(access_key_to_use, SecretStr):
            access_key_to_use = access_key_to_use.get_secret_value()

        try:
            self._client = Minio(
                endpoint=endpoint_to_use,
                access_key=access_key_to_use,
                secret_key=secret_key_to_use,
                secure=secure_flag,
                region=region_to_use,
            )
        except Exception as exc:
            log.critical(
                "Failed to initialize MinIO client.",
                error=str(exc),
                endpoint=endpoint_to_use,
            )
            raise MinioIndexStorageError(
                f"Failed to initialize MinIO client for endpoint '{endpoint_to_use}': {exc}"
            ) from exc

        try:
            if not self._client.bucket_exists(self.bucket_name):
                self._client.make_bucket(self.bucket_name, location=region_to_use)
        except S3Error as exc:
            log.critical(
                "Failed to ensure MinIO bucket existence.",
                bucket=self.bucket_name,
                error=str(exc),
            )
            raise MinioIndexStorageError(
                f"Failed to ensure MinIO bucket '{self.bucket_name}' exists: {exc}"
            ) from exc
        except Exception as exc:
            log.critical(
                "Unexpected error while ensuring MinIO bucket.",
                bucket=self.bucket_name,
                error=str(exc),
                exc_info=True,
            )
            raise MinioIndexStorageError(
                f"Unexpected error ensuring bucket '{self.bucket_name}': {exc}"
            ) from exc

        self.secure = secure_flag
        self.region = region_to_use
        self.log = log.bind(
            bucket=self.bucket_name,
            endpoint=endpoint_to_use,
            adapter="MinioIndexStorageAdapter",
        )

    def _object_path(self, company_id: uuid.UUID, filename: str) -> str:
        return f"{INDEX_ROOT_PATH}/{company_id}/{filename}"

    async def load_index_files(self, company_id: uuid.UUID) -> Tuple[Optional[str], Optional[str]]:
        adapter_log = self.log.bind(company_id=str(company_id), action="load_index_files")
        adapter_log.info("Attempting to load index files from MinIO.")

        temp_dir = Path(tempfile.mkdtemp(prefix=f"sparse_idx_{company_id}_"))
        local_bm2s_path = temp_dir / BM25_DUMP_FILENAME
        local_id_map_path = temp_dir / ID_MAP_FILENAME

        object_bm25 = self._object_path(company_id, BM25_DUMP_FILENAME)
        object_id_map = self._object_path(company_id, ID_MAP_FILENAME)

        loop = asyncio.get_running_loop()

        async def _download(object_name: str, destination: Path) -> bool:
            try:
                await loop.run_in_executor(
                    None,
                    self._client.fget_object,
                    self.bucket_name,
                    object_name,
                    str(destination),
                )
                adapter_log.debug(
                    "Downloaded object from MinIO.",
                    object_name=object_name,
                    local_file=str(destination),
                )
                return True
            except S3Error as exc:
                if exc.code == "NoSuchKey":
                    adapter_log.warning(
                        "Object not found in MinIO.",
                        object_name=object_name,
                    )
                else:
                    adapter_log.error(
                        "MinIO error while downloading object.",
                        object_name=object_name,
                        error=str(exc),
                    )
                return False
            except Exception as exc:
                adapter_log.exception(
                    "Unexpected error downloading object from MinIO.",
                    object_name=object_name,
                )
                return False

        bm25_downloaded = await _download(object_bm25, local_bm2s_path)
        id_map_downloaded = await _download(object_id_map, local_id_map_path)

        if bm25_downloaded and id_map_downloaded:
            adapter_log.info("Both index files downloaded successfully from MinIO.")
            return str(local_bm2s_path), str(local_id_map_path)

        adapter_log.warning(
            "Failed to download one or both index files from MinIO. Cleaning up temporary files.")
        for file_path in (local_bm2s_path, local_id_map_path):
            try:
                if file_path.exists():
                    file_path.unlink()
            except OSError as exc:
                adapter_log.error(
                    "Unable to remove temporary file after failed download.",
                    file=str(file_path),
                    error=str(exc),
                )
        try:
            temp_dir.rmdir()
        except OSError:
            pass
        return None, None

    async def save_index_files(
        self,
        company_id: uuid.UUID,
        local_bm2s_dump_path: str,
        local_id_map_path: str,
    ) -> None:
        adapter_log = self.log.bind(company_id=str(company_id), action="save_index_files")
        adapter_log.info("Attempting to upload index files to MinIO.")

        object_bm25 = self._object_path(company_id, BM25_DUMP_FILENAME)
        object_id_map = self._object_path(company_id, ID_MAP_FILENAME)

        loop = asyncio.get_running_loop()

        async def _upload(local_path: str, object_name: str, content_type: str) -> None:
            try:
                await loop.run_in_executor(
                    None,
                    self._client.fput_object,
                    self.bucket_name,
                    object_name,
                    local_path,
                    content_type,
                )
                adapter_log.debug(
                    "Uploaded file to MinIO.",
                    object_name=object_name,
                    local_file=local_path,
                )
            except S3Error as exc:
                adapter_log.error(
                    "MinIO error while uploading object.",
                    object_name=object_name,
                    error=str(exc),
                )
                raise MinioIndexStorageError(
                    f"MinIO error uploading {local_path} to {object_name}: {exc}"
                ) from exc
            except Exception as exc:
                adapter_log.exception(
                    "Unexpected error uploading object to MinIO.",
                    object_name=object_name,
                )
                raise MinioIndexStorageError(
                    f"Unexpected error uploading {local_path} to {object_name}: {exc}"
                ) from exc

        await _upload(local_bm2s_dump_path, object_bm25, "application/octet-stream")
        await _upload(local_id_map_path, object_id_map, "application/json")
        adapter_log.info("Both index files uploaded successfully to MinIO.")