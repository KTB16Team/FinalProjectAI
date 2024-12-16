from openai import OpenAI
import boto3
import os
import tempfile # 임시 파일과 임시 디렉터리를 생성하고 관리하는 Python 표준 라이브러리
from botocore.exceptions import ClientError, NoCredentialsError
from core.config import settings
class S3SttService:
    def __init__(self):
        stt_service = S3SttService(
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
            bucket_name = settings.AWS_S3_BUCKET_NAME,
            region_name=settings.AWS_REGION,
            openai_api_key=settings.OPENAI_API_KEY,
        )
    async def download_file_from_s3(self, bucket_name, object_key):
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False) # delete=False-> 다운로드 작업이 끝날 때까지 파일을 유지.
            self.s3_client.download_file(bucket_name, object_key, temp_file.name)
            return temp_file.name
        except ClientError as e:
            print(f"Error downloading file from S3: {e}")
            raise

    async def transcribe_audio(self, file_path):
        try:
            with open(file_path, "rb") as audio_file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return transcription.text
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            raise
                
    async def process_audio_file(self, bucket_name, object_key):
        try:
            temp_file_path = self.download_file_from_s3(bucket_name, object_key)
            transcription = self.transcribe_audio(temp_file_path)
            os.unlink(temp_file_path) # temporary file clean up = 임시 파일이 더 이상 필요하지 않을 때(작업 완료 시) 초기화(자동 삭제됨)
            return transcription
        except Exception as e:
            print(f"Error processing audio file: {e}")
            raise


if __name__ == "__main__":
    AWS_ACCESS_KEY_ID = ""
    AWS_SECRET_ACCESS_KEY = ""
    REGION_NAME = ""
    OPENAI_API_KEY = ""
    stt_service = S3SttService(
        aws_access_key_id = AWS_ACCESS_KEY_ID,
        aws_secret_access_key= AWS_SECRET_ACCESS_KEY,
        region_name = REGION_NAME,
        openai_api_key=OPENAI_API_KEY
    )

    try:
        transcription = stt_service.process_audio_file(
            bucket_name="",
            object_key="/Users/alice.kim/Desktop/aa/Final/app/services/SttTest.m4a"
        )
        print("Transcription:", transcription)
    except Exception as e:
        print(f"Failed to process audio file: {e}")

# whisperapi stt code
# client = OpenAI(api_key="")
# audio_file = open('{파일위치}.mp3', "rb")
# transcription = client.audio.transcriptions.create(
#     model="whisper-1",
#     file=audio_file
# )
# print(transcription.text)

######

def parse_s3_url(url):
    """
    S3 URL에서 bucket_name과 object_key를 추출하는 함수
    """
    parsed_url = urlparse(url)
    bucket_name = parsed_url.netloc.split(".")[0]  # 'ktb-aimo'
    object_key = parsed_url.path.lstrip("/")  # 'audio/SttTest.m4a'
    return bucket_name, object_key

def download_s3_file(url, aws_access_key, aws_secret_key, region_name):
    """
    S3 파일 다운로드 함수
    """
    try:
        # URL에서 bucket_name과 object_key 추출
        bucket_name, object_key = parse_s3_url(url)

        # S3 클라이언트 생성
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name,
        )

        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(delete=False)

        # S3에서 파일 다운로드
        s3_client.download_file(bucket_name, object_key, temp_file.name)
        print(f"File downloaded to: {temp_file.name}")
        return temp_file.name
    except ClientError as e:
        print(f"Error downloading file from S3: {e}")
        raise

# 사용 예제
if __name__ == "__main__":
    s3_url = "https://ktb-aimo.s3.ap-northeast-2.amazonaws.com/audio/SttTest.m4a"
    aws_access_key = "your_access_key"
    aws_secret_key = "your_secret_key"
    region_name = "ap-northeast-2"

    downloaded_file_path = download_s3_file(s3_url, aws_access_key, aws_secret_key, region_name)
    print(f"Downloaded file saved at: {downloaded_file_path}")