import os
from core.config import settings
from openai import OpenAI
from services.download_s3_file import download_s3_file
from core.logging import logger

async def transcribe_audio(file_path):
    try:
        # 파일 확장자 검사
        valid_extensions = ['.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.wav', '.webm']
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if not file_ext:
            raise ValueError("File has no extension")
        if file_ext not in valid_extensions:
            raise ValueError(f"Unsupported file format. Supported formats: {valid_extensions}")

        # 파일 존재 여부 및 크기 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError("File is empty")

        logger.info(f"Processing audio file: {file_path}, Size: {file_size} bytes, Format: {file_ext}")

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        logger.info("Audio transcription completed successfully")
        return transcription.text

    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise

async def process_audio_file(url):
    temp_file_path = None
    try:
        # 임시 파일 다운로드
        temp_file_path = await download_s3_file(url)
        logger.info(f"File downloaded successfully: {temp_file_path}")
        
        # 파일 정보 로깅
        file_size = os.path.getsize(temp_file_path)
        file_ext = os.path.splitext(temp_file_path)[1]
        logger.info(f"File details - Size: {file_size} bytes, Extension: {file_ext}")
        
        # 음성 파일 변환
        transcription = await transcribe_audio(temp_file_path)
        return transcription

    except Exception as e:
        logger.error(f"Error processing audio file: {e}", exc_info=True)
        raise
    finally:
        # 임시 파일 삭제
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Temporary file removed: {temp_file_path}")
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")