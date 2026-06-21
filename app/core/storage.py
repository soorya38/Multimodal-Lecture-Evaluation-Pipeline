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
