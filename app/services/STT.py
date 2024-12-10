from openai import OpenAI
import boto3
import os
import tempfile # 임시 파일과 임시 디렉터리를 생성하고 관리하는 Python 표준 라이브러리
from botocore.exceptions import ClientError

class S3SttService:
    def __init__(self, aws_access_key_id, aws_secret_access_key, region_name, openai_api_key):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id = aws_access_key_id,
            aws_secret_access_key= aws_secret_access_key,
            region_name = region_name # 진짜 말그대로 지역
        )
        self.openai_client = OpenAI(api_key='')
    def download_file_from_s3(self, bucket_name, object_key):
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False) # delete=False-> 다운로드 작업이 끝날 때까지 파일을 유지.
            self.s3_client.download_file(bucket_name, object_key, temp_file.name)
            return temp_file.name
        except ClientError as e:
            print(f"Error downloading file from S3: {e}")
            raise

    def transcribe_audio(self, file_path):
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
                
    def process_audio_file(self, bucket_name, object_key):
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