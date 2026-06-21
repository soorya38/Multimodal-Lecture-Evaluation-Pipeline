import os
import structlog
from minio import Minio
from minio.error import S3Error

logger = structlog.get_logger(__name__)

# Global MinIO client instance
minio_client: Minio | None = None

def init_minio() -> None:
    """
    Initialize the MinIO client and ensure default buckets exist.
    """
    global minio_client
    
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    default_bucket = os.getenv("MINIO_DEFAULT_BUCKET", "lectures")
    
    logger.info("Initializing MinIO client", endpoint=endpoint, secure=secure, bucket=default_bucket)
    
    try:
        minio_client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        
        # Check if the default bucket exists, create it if it doesn't
        found = minio_client.bucket_exists(default_bucket)
        if not found:
            minio_client.make_bucket(default_bucket)
            logger.info("Created default MinIO bucket", bucket=default_bucket)
        else:
            logger.info("Default MinIO bucket already exists", bucket=default_bucket)
            
    except S3Error as e:
        logger.error("Failed to initialize MinIO bucket", error=str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error initializing MinIO", error=str(e))
        raise

def get_minio_client() -> Minio:
    """
    Get the initialized MinIO client instance.
    Raises an error if called before initialization.
    """
    global minio_client
    if minio_client is None:
        raise RuntimeError("MinIO client has not been initialized. Call init_minio() first.")
    return minio_client


def upload_file(
    bucket: str,
    object_name: str,
    file_path: str,
    content_type: str,
) -> None:
    """
    Upload a local file to a MinIO bucket.

    Args:
        bucket: Target bucket name.
        object_name: The key / path under which the object will be stored.
        file_path: Absolute path to the local file to upload.
        content_type: MIME type for the uploaded object (e.g. "video/mp4").
    """
    client = get_minio_client()

    logger.info(
        "Uploading file to MinIO",
        bucket=bucket,
        object_name=object_name,
        file_path=file_path,
        content_type=content_type,
    )

    try:
        client.fput_object(
            bucket,
            object_name,
            file_path,
            content_type=content_type,
        )
        logger.info(
            "Successfully uploaded file to MinIO",
            bucket=bucket,
            object_name=object_name,
        )
    except S3Error as e:
        logger.error(
            "Failed to upload file to MinIO",
            bucket=bucket,
            object_name=object_name,
            error=str(e),
        )
        raise


def download_file(
    bucket: str,
    object_name: str,
    file_path: str,
) -> None:
    """
    Download an object from a MinIO bucket to a local file path.

    Args:
        bucket: Source bucket name.
        object_name: The key / path of the object to download.
        file_path: Local destination path where the file will be written.
    """
    client = get_minio_client()

    logger.info(
        "Downloading file from MinIO",
        bucket=bucket,
        object_name=object_name,
        file_path=file_path,
    )

    try:
        client.fget_object(bucket, object_name, file_path)
        logger.info(
            "Successfully downloaded file from MinIO",
            bucket=bucket,
            object_name=object_name,
        )
    except S3Error as e:
        logger.error(
            "Failed to download file from MinIO",
            bucket=bucket,
            object_name=object_name,
            error=str(e),
        )
        raise


def list_objects(bucket: str, prefix: str) -> list[str]:
    """
    List all object keys in a MinIO bucket that match a specific prefix.

    Args:
        bucket: The MinIO bucket name.
        prefix: The prefix to search for (e.g., "1234abcd/frames/").

    Returns:
        A list of object keys.
    """
    client = get_minio_client()

    logger.info("Listing objects in MinIO", bucket=bucket, prefix=prefix)
    
    try:
        objects = client.list_objects(bucket_name=bucket, prefix=prefix, recursive=True)
        keys = [obj.object_name for obj in objects if obj.object_name]
        logger.info("Listed objects", bucket=bucket, prefix=prefix, count=len(keys))
        return keys
    except S3Error as e:
        logger.error(
            "Failed to list objects in MinIO",
            bucket=bucket,
            prefix=prefix,
            error=str(e),
        )
        raise
