import boto3
import os
import tempfile
from urllib.parse import urlparse, unquote
from botocore.exceptions import ClientError
from core.config import settings
from core.logging import logger

async def parse_s3_url(url):
    parsed_url = urlparse(url)
    bucket_name = parsed_url.netloc.split(".")[0]
    # URL 디코딩 및 '+'를 공백으로 변환
    object_key = unquote(parsed_url.path.lstrip("/")).replace("+", " ")
    
    # 객체 키에서 원본 파일명 추출
    filename = object_key.split('/')[-1]
    return bucket_name, object_key, filename

async def download_s3_file(url):
    bucket_name, object_key, filename = await parse_s3_url(url)
    logger.info(f"bucket: {bucket_name}, key: {object_key}")

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY,
        aws_secret_access_key=settings.AWS_SECRET_KEY,
        region_name=settings.AWS_REGION,
    )

    try:
        # 객체 메타데이터를 통해 ContentType 확인
        response = s3_client.head_object(Bucket=settings.AWS_S3_BUCKET_NAME, Key=object_key)
        content_type = response.get('ContentType', '')
        logger.info(f"File content type: {content_type}")

        # 콘텐츠 타입에 따른 확장자 매핑
        content_type_to_ext = {
            'audio/mpeg': '.mp3',
            'audio/mp3': '.mp3',
            'audio/wav': '.wav',
            'audio/wave': '.wav',
            'audio/x-wav': '.wav',
            'audio/ogg': '.ogg',
            'audio/flac': '.flac',
            'audio/x-m4a': '.m4a',
            'audio/mp4': '.m4a',
            'audio/x-mpg': '.mpga',
            'audio/mpeg3': '.mp3',
            'audio/x-mpeg-3': '.mp3'
        }

        # temp 디렉토리를 명시적으로 지정
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        # 파일 확장자 결정
        if '.' not in filename:
            for mime_type, ext in content_type_to_ext.items():
                if mime_type.lower() in content_type.lower():
                    filename = filename + ext
                    logger.info(f"Added extension {ext} based on content type")
                    break
            else:
                # 기본값으로 .mp3 설정
                filename = filename + '.mp3'
                logger.info("No matching content type found, using default .mp3 extension")

        temp_file_path = os.path.join(temp_dir, filename)
        logger.info(f"Temporary file path: {temp_file_path}")

        # 파일 다운로드
        logger.info(f"Downloading from bucket: {bucket_name}, key: {object_key}")
        s3_client.download_file(bucket_name, object_key, temp_file_path)
        logger.info(f"File downloaded to: {temp_file_path}")

        # 파일 크기 및 존재 여부 확인
        if os.path.exists(temp_file_path):
            file_size = os.path.getsize(temp_file_path)
            logger.info(f"Downloaded file size: {file_size} bytes")
        else:
            raise FileNotFoundError("Downloaded file not found at the specified path")

        return temp_file_path

    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            logger.error(f"Error: Object '{object_key}' not found in bucket '{bucket_name}'.")
        else:
            logger.error(f"Unexpected S3 error: {e}")
        raise
    except Exception as e:
        logger.error(f"General error during file download: {e}")
        raise