from openai import OpenAI
from urllib.parse import urlparse, unquote
import boto3
import os
import tempfile # 임시 파일과 임시 디렉터리를 생성하고 관리하는 Python 표준 라이브러리
from botocore.exceptions import ClientError, NoCredentialsError
from core.config import settings
# class S3SttService:
#     def __init__(self):
#         stt_service = S3SttService(
#             aws_access_key_id=settings.AWS_ACCESS_KEY,
#             aws_secret_access_key=settings.AWS_SECRET_KEY,
#             bucket_name = settings.AWS_S3_BUCKET_NAME,
#             region_name=settings.AWS_REGION,
#             openai_api_key=settings.OPENAI_API_KEY,
#         )
#     # async def download_file_from_s3(self, bucket_name, object_key):
#     #     try:
#     #         temp_file = tempfile.NamedTemporaryFile(delete=False) # delete=False-> 다운로드 작업이 끝날 때까지 파일을 유지.
#     #         self.s3_client.download_file(bucket_name, object_key, temp_file.name)
#     #         return temp_file.name
#     #     except ClientError as e:
#     #         print(f"Error downloading file from S3: {e}")
#     #         raise
#
#     async def transcribe_audio(self, file_path):
#         try:
#             with open(file_path, "rb") as audio_file:
#                 transcription = self.openai_client.audio.transcriptions.create(
#                     model="whisper-1",
#                     file=audio_file
#                 )
#             return transcription.text
#         except Exception as e:
#             print(f"Error transcribing audio: {e}")
#             raise
#
#     async def process_audio_file(self, bucket_name, object_key):
#         try:
#             temp_file_path = self.download_file_from_s3(bucket_name, object_key)
#             transcription = self.transcribe_audio(temp_file_path)
#             os.unlink(temp_file_path) # temporary file clean up = 임시 파일이 더 이상 필요하지 않을 때(작업 완료 시) 초기화(자동 삭제됨)
#             return transcription
#         except Exception as e:
#             print(f"Error processing audio file: {e}")
#             raise
#
#
# if __name__ == "__main__":
#     AWS_ACCESS_KEY_ID = ""
#     AWS_SECRET_ACCESS_KEY = ""
#     REGION_NAME = ""
#     OPENAI_API_KEY = ""
#     stt_service = S3SttService(
#         aws_access_key_id = AWS_ACCESS_KEY_ID,
#         aws_secret_access_key= AWS_SECRET_ACCESS_KEY,
#         region_name = REGION_NAME,
#         openai_api_key=OPENAI_API_KEY
#     )
#
#     try:
#         transcription = stt_service.process_audio_file(
#             bucket_name="",
#             object_key="/Users/alice.kim/Desktop/aa/Final/app/services/SttTest.m4a"
#         )
#         print("Transcription:", transcription)
#     except Exception as e:
#         print(f"Failed to process audio file: {e}")

# whisperapi stt code
# client = OpenAI(api_key="")
# audio_file = open('{파일위치}.mp3', "rb")
# transcription = client.audio.transcriptions.create(
#     model="whisper-1",
#     file=audio_file
# )
# print(transcription.text)

######
# class S3SttService:
import boto3
import tempfile
from urllib.parse import urlparse, unquote
from botocore.exceptions import ClientError
from core.config import settings
from core.logging import logger
def parse_s3_url(url):
    try:
        parsed_url = urlparse(url)
        bucket_name = parsed_url.netloc.split(".")[0]
        # URL 디코딩 및 '+'를 공백으로 변환
        object_key = unquote(parsed_url.path.lstrip("/")).replace("+", " ")
        return bucket_name, object_key
    except Exception as e:
        logger.error(f"Error parsing S3 URL: {e}")

def download_s3_file(url, download_dir="temp"):
    # 다운로드 디렉토리 설정
    os.makedirs(download_dir, exist_ok=True)  # 디렉토리가 없으면 생성

    bucket_name, object_key = parse_s3_url(url)
    print(f"bucket: {bucket_name}, key: {object_key}")
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
            region_name=settings.AWS_REGION,
        )
        logger.info("S3 Client initialized successfully.")

        # 객체 존재 확인
        s3_client.head_object(Bucket=settings.AWS_S3_BUCKET_NAME, Key=object_key)
        logger.info(f"File exists in S3. Proceeding with download: {object_key}")

        # 로컬 파일 경로 생성
        file_name = os.path.basename(object_key)
        local_file_path = os.path.join(download_dir, file_name)

        # 파일 다운로드
        # print(f"Downloading from bucket: {bucket_name}, key: {object_key}")
        s3_client.download_file(bucket_name, object_key, local_file_path)
        logger.info(f"File downloaded successfully to: {local_file_path}")
        # print(f"File downloaded to: {local_file_path}")
        return local_file_path
    except ClientError as e:
        logger.error(f"S3 ClientError: {e.response['Error']['Message']}")
        raise
    except Exception as e:
        logger.error(f"General error during S3 download: {e}")
        raise
        
    # except ClientError as e:
    #     if e.response['Error']['Code'] == "404":
    #         print(f"Error: Object '{object_key}' not found in bucket '{bucket_name}'.")
    #     else:
    #         print(f"Unexpected error: {e}")
    #     raise
    # except Exception as e:
    #     print(f"General error: {e}")
    #     raise



    # 사용 예제
    # if __name__ == "__main__":
    #     s3_url = "https://ktb-aimo.s3.ap-northeast-2.amazonaws.com/audio/SttTest.m4a"
    #     aws_access_key = "your_access_key"
    #     aws_secret_key = "your_secret_key"
    #     region_name = "ap-northeast-2"
    #
    #     downloaded_file_path = download_s3_file(s3_url, aws_access_key, aws_secret_key, region_name)
    #     print(f"Downloaded file saved at: {downloaded_file_path}")