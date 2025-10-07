# ingest-service/app/services/minio_client.py
import structlog
import asyncio
from typing import Optional
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, BotoCoreError
from app.core.config import settings

log = structlog.get_logger(__name__)

class MinIOClientError(Exception):
    """Custom exception for MinIO related errors."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        self.message = message
        self.original_exception = original_exception
        super().__init__(message)

    def __str__(self):
        if self.original_exception:
            return f"{self.message}: {type(self.original_exception).__name__} - {str(self.original_exception)}"
        return self.message

class MinIOClient:
    """Client to interact with a MinIO S3-compatible storage service."""
    def __init__(self, bucket_name: Optional[str] = None):
        self.bucket_name = bucket_name or settings.MINIO_BUCKET_NAME
        self.endpoint_url = f"http{'s' if settings.MINIO_SECURE else ''}://{settings.MINIO_ENDPOINT}"
        self.log = log.bind(minio_bucket=self.bucket_name, minio_endpoint=self.endpoint_url)
        
        try:
            self._client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=settings.MINIO_ACCESS_KEY.get_secret_value(),
                aws_secret_access_key=settings.MINIO_SECRET_KEY.get_secret_value(),
                config=Config(signature_version='s3v4'),
                region_name='us-east-1' 
            )
            self._ensure_bucket_exists()
        except (BotoCoreError, ClientError) as e:
            self.log.error("Failed to initialize MinIO client", error=str(e))
            raise MinIOClientError("Could not initialize MinIO client", e) from e
        
    def _ensure_bucket_exists(self):
        """Checks if the bucket exists and creates it if not."""
        try:
            self._client.head_bucket(Bucket=self.bucket_name)
            self.log.info("MinIO bucket already exists.", bucket_name=self.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                self.log.warning("MinIO bucket does not exist. Creating it...", bucket_name=self.bucket_name)
                try:
                    self._client.create_bucket(Bucket=self.bucket_name)
                    self.log.info("Successfully created MinIO bucket.", bucket_name=self.bucket_name)
                except ClientError as create_e:
                    self.log.error("Failed to create MinIO bucket.", error=str(create_e))
                    raise MinIOClientError(f"Failed to create bucket {self.bucket_name}", create_e) from create_e
            else:
                self.log.error("Error checking for MinIO bucket.", error=str(e))
                raise MinIOClientError(f"Error checking bucket {self.bucket_name}", e) from e

    async def upload_file_async(self, object_name: str, data: bytes, content_type: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.upload_file_sync, object_name, data, content_type)

    async def download_file_async(self, object_name: str, file_path: str):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.download_file_sync, object_name, file_path)

    async def check_file_exists_async(self, object_name: str) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.check_file_exists_sync, object_name)

    async def delete_file_async(self, object_name: str):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.delete_file_sync, object_name)

    # --- Synchronous methods for worker compatibility ---
    
    def upload_file_sync(self, object_name: str, data: bytes, content_type: str) -> str:
        self.log.info("Uploading file to MinIO (sync)...", object_name=object_name, content_type=content_type, length=len(data))
        try:
            self._client.put_object(
                Bucket=self.bucket_name,
                Key=object_name,
                Body=data,
                ContentType=content_type
            )
            self.log.info("File uploaded successfully to MinIO (sync)", object_name=object_name)
            return object_name
        except (ClientError, BotoCoreError) as e:
            self.log.error("MinIO upload failed (sync)", error=str(e))
            raise MinIOClientError(f"MinIO error uploading {object_name}", e) from e

    def download_file_sync(self, object_name: str, file_path: str):
        self.log.info("Downloading file from MinIO (sync)...", object_name=object_name, target_path=file_path)
        try:
            self._client.download_file(self.bucket_name, object_name, file_path)
            self.log.info("File downloaded successfully from MinIO (sync)", object_name=object_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                self.log.error("Object not found in MinIO (sync)", object_name=object_name)
                raise MinIOClientError(f"Object not found in MinIO: {object_name}", e) from e
            else:
                self.log.error("MinIO download failed (sync)", error=str(e))
                raise MinIOClientError(f"MinIO error downloading {object_name}", e) from e

    def check_file_exists_sync(self, object_name: str) -> bool:
        self.log.debug("Checking file existence in MinIO (sync)", object_name=object_name)
        try:
            self._client.head_object(Bucket=self.bucket_name, Key=object_name)
            self.log.debug("File existence check completed in MinIO (sync)", object_name=object_name, exists=True)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                self.log.debug("File does not exist in MinIO (sync)", object_name=object_name)
                return False
            self.log.exception("Unexpected error during MinIO existence check (sync)", error=str(e))
            return False
        except Exception:
            self.log.exception("Unexpected Boto3 error during MinIO existence check (sync)")
            return False

    def delete_file_sync(self, object_name: str):
        self.log.info("Deleting file from MinIO (sync)...", object_name=object_name)
        try:
            self._client.delete_object(Bucket=self.bucket_name, Key=object_name)
            self.log.info("File deleted successfully from MinIO (sync)", object_name=object_name)
        except (ClientError, BotoCoreError) as e:
            self.log.error("MinIO delete failed (sync)", error=str(e))
            raise MinIOClientError(f"MinIO error deleting {object_name}", e) from e