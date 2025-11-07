import logging
from typing import Optional
import boto3
from botocore.exceptions import ClientError
from app.core.config import settings

logger = logging.getLogger(__name__)


class S3Client:
    def __init__(self):
        self.s3_client = boto3.client("s3", region_name=settings.aws_region)
        self.bucket = settings.s3_bucket

    def upload_file(self, file_obj, s3_key: str, content_type: str) -> bool:
        """Upload file to S3."""
        try:
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket,
                s3_key,
                ExtraArgs={"ContentType": content_type},
            )
            logger.info(f"Uploaded file to s3://{self.bucket}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Error uploading to S3: {e}")
            return False

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download file from S3 to local path."""
        try:
            self.s3_client.download_file(self.bucket, s3_key, local_path)
            logger.info(f"Downloaded file from s3://{self.bucket}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Error downloading from S3: {e}")
            return False

    def generate_presigned_url(
        self, s3_key: str, expiration: Optional[int] = None
    ) -> Optional[str]:
        """Generate presigned URL for S3 object."""
        if expiration is None:
            expiration = settings.presigned_url_ttl
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": s3_key},
                ExpiresIn=expiration,
            )
            return url
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None


s3_client = S3Client()

