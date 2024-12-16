import os
from core.config import settings
from openai import OpenAI
from services.download_s3_file import download_s3_file
async def transcribe_audio(file_path):
    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcription.text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise

async def process_audio_file(url):
    try:
        # 임시 파일 다운로드
        temp_file_path = await download_s3_file(url)
        # 음성 파일 변환
        transcription = await transcribe_audio(temp_file_path)
        # 임시 파일 삭제
        os.unlink(temp_file_path)
        return transcription
    except Exception as e:
        print(f"Error processing audio file: {e}")
        raise