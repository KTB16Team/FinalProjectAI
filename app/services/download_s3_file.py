import boto3
import os
import tempfile
from urllib.parse import urlparse, unquote
from botocore.exceptions import ClientError
from core.config import settings


async def parse_s3_url(url):
    parsed_url = urlparse(url)
    bucket_name = parsed_url.netloc.split(".")[0]
    # URL 디코딩 및 '+'를 공백으로 변환
    object_key = unquote(parsed_url.path.lstrip("/")).replace("+", " ")
    return bucket_name, object_key

async def download_s3_file(url):
    bucket_name, object_key = await parse_s3_url(url)
    print(f"bucket: {bucket_name}, key: {object_key}")

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY,
        aws_secret_access_key=settings.AWS_SECRET_KEY,
        region_name=settings.AWS_REGION,
    )

    try:
        # 객체 존재 확인
        s3_client.head_object(Bucket=settings.AWS_S3_BUCKET_NAME, Key=object_key)
        print(f"Object '{object_key}' exists. Proceeding with download...")

        # temp 디렉토리를 명시적으로 지정
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        # 파일 확장자 유지하여 임시 파일 경로 생성
        file_extension = os.path.splitext(object_key)[1]  # 확장자 추출
        with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=file_extension, delete=False) as temp_file:
            temp_file_path = temp_file.name

        # 파일 다운로드
        print(f"Downloading from bucket: {bucket_name}, key: {object_key}")
        s3_client.download_file(bucket_name, object_key, temp_file_path)
        print(f"File downloaded to: {temp_file_path}")
        return temp_file_path
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"Error: Object '{object_key}' not found in bucket '{bucket_name}'.")
        else:
            print(f"Unexpected error: {e}")
        raise
    except Exception as e:
        print(f"General error: {e}")
        raise
