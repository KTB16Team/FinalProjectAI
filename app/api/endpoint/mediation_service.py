import json
import ssl
from fastapi import APIRouter, HTTPException, Header, status, BackgroundTasks
from botocore.exceptions import ClientError
from datetime import datetime
from models.info import DataInfoSummary, VoiceInfo, DataInfoSTT, JudgeRequest, STTRequest, ConflictAnalysisRequest, ConflictAnalysisResponseData, ConflictAnalysisResponse
from services.situation_summary import situation_summary_GPT, stt_model, generate_response, test_response
from services.audio_process import process_audio_file
from services.image_process import process_image_file
from core.logging import logger
from core.config import settings
import requests
import pika
# from services.score_multi import ConflictAnalyzer
import torch
import os
import httpx

router = APIRouter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@router.post("/analyze-conflict", response_model=ConflictAnalysisResponseData, status_code=201)
async def analyze_conflict(request: ConflictAnalysisRequest):
    logger.info(f"Received conflict analysis request: {request.dict()}")

    if not request.content:
        logger.error("CONTENT_NOT_PROVIDED")
        raise HTTPException(status_code=400, detail="CONTENT_NOT_PROVIDED")

    try:
        # ConflictAnalyzer 사용 부분은 주석 상태이므로 실제 구현 시 참고
        analysis_result = await conflict_analyzer.analyze_content(
            content=request.content,
            request_id=request.id
        )

        if analysis_result["status"] != "success":
            logger.error(f"Conflict analysis failed: {analysis_result.get('message', 'Unknown error')}")
            raise HTTPException(status_code=500, detail="Conflict analysis failed")

        backend_payload = analysis_result['data']
        backend_server_url = os.getenv("BACKEND_SERVER_URL", "https://api.ktb-aimo.link/api/v1/private-posts/judgement/callback")

        async with httpx.AsyncClient() as client:
            backend_response = await client.post(backend_server_url, json=backend_payload)

            if backend_response.status_code != 200:
                logger.error(f"Failed to send data to backend server: {backend_response.status_code}")
                raise HTTPException(status_code=502, detail="Failed to send data to backend server")

        response = ConflictAnalysisResponse(
            status=analysis_result["status"],
            method=analysis_result["method"],
            data=ConflictAnalysisResponseData(**analysis_result["data"])
        )

        logger.info(f"Conflict analysis successful: {response}")
        return response

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error when communicating with backend server: {e}")
        raise HTTPException(status_code=502, detail="Backend server error")
    except httpx.RequestError as e:
        logger.error(f"Request error when communicating with backend server: {e}")
        raise HTTPException(status_code=503, detail="Backend server connection error")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=500, detail="Invalid JSON response from ConflictAnalyzer")
    except Exception as e:
        logger.error(f"Unexpected error during conflict analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during conflict analysis")

@router.post("/speech-to-text", response_model=VoiceInfo, status_code=201)
async def get_voice(request: STTRequest):
    logger.info("get_infos start")
    logger.info(f"audio URL : {request.url}")

    if not request.url:
        raise HTTPException(status_code=400, detail="URL_NOT_PROVIDED")

    try:
        transcription = await process_audio_file(request.url)
        logger.info("STT processing completed successfully.")
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            logger.error("File not found in S3.")
            raise HTTPException(status_code=404, detail="FILE_NOT_FOUND_IN_S3")
        else:
            logger.error(f"Unexpected S3 error: {e}")
            raise HTTPException(status_code=500, detail="S3_ERROR")
    except Exception as e:
        logger.error(f"General STT processing error: {e}")
        raise HTTPException(status_code=500, detail="STT_PROCESSING_ERROR")

    response = VoiceInfo(
        status="Created",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data=DataInfoSTT(
            script=transcription
        )
    )
    logger.info(f"Response: {response}")
    return response

@router.post("/image-to-text", response_model=VoiceInfo, status_code=201)
async def get_image(request: STTRequest):
    logger.info("get_infos start")
    logger.info(f"image URL : {request.url}")

    if not request.url:
        raise HTTPException(status_code=400, detail="URL_NOT_PROVIDED")

    try:
        transcription = await process_image_file(request.url)
        logger.info(transcription)
        logger.info("OCR processing completed successfully.")
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            logger.error("File not found in S3.")
            raise HTTPException(status_code=404, detail="FILE_NOT_FOUND_IN_S3")
        else:
            logger.error(f"Unexpected S3 error: {e}")
            raise HTTPException(status_code=500, detail="S3_ERROR")
    except Exception as e:
        logger.error(f"General OCR processing error: {e}")
        raise HTTPException(status_code=500, detail="OCR_PROCESSING_ERROR")

    response = VoiceInfo(
        status="Created",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data=DataInfoSTT(
            script=transcription
        )
    )
    logger.info(f"Response: {response}")
    return response

# 기존 비동기 로직
CALLBACK_URL = settings.CALLBACK_URL
ACCESSTOKEN = settings.ACCESSTOKEN

@router.post("/judgement", status_code=202)
async def process_judge(request: JudgeRequest, background_tasks: BackgroundTasks):
    logger.info(f"Received request data: {request.dict()}")

    if not request.content:
        raise HTTPException(status_code=400, detail="CONTENT_NOT_PROVIDED")

    # Background 작업 등록
    logger.info("Starting background task for judgement processing...")
    background_tasks.add_task(execute_test_response_and_callback, request.content, request.id)

    return {"status": "accepted", "message": "Judgement processing started."}

def execute_test_response_and_callback(content: str, id: int):
    """
    test_response 호출 후 결과를 CALLBACK_URL로 전송
    """
    try:
        logger.info("Executing test_response function...")
        entities = test_response(content)

        # 필수 필드 카멜케이스로 수정
        required_fields = [
            "title", "stancePlaintiff", "stanceDefendant",
            "summaryAi", "judgement", "faultRate"
        ]
        missing_fields = [field for field in required_fields if field not in entities]

        if missing_fields:
            logger.error(f"Missing fields in GPT response: {missing_fields}")
            DataInfoSummaryJson = {
                "status": False,
                "id": id
            }
        else:
            DataInfoSummaryJson = {
                "status": True,
                "accessKey": ACCESSTOKEN,
                "id": id,
                "title": entities.get("title"),
                "stancePlaintiff": entities.get("stancePlaintiff"),
                "stanceDefendant": entities.get("stanceDefendant"),
                "summaryAi": entities.get("summaryAi"),
                "judgement": entities.get("judgement"),
                "faultRate": entities.get("faultRate")
            }

        logger.info(f"Sending POST request to DataInfoSummary: {DataInfoSummaryJson}")
        logger.info(f"Sending POST request to CALLBACK_URL: {CALLBACK_URL}")
        response = requests.post(CALLBACK_URL, json=DataInfoSummaryJson)
        logger.info(f"Callback response: {response.status_code}, {response.text}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        error_response = {"status": "error", "message": str(e)}
        requests.post(CALLBACK_URL, json=error_response)

def process_message(ch, method, properties, body):
    """
    RabbitMQ 메시지 소비 후 처리
    """
    try:
        decoded_body = body.decode('utf-8')
        logger.info(f"Received message: {decoded_body}")
        message = json.loads(decoded_body)
        content = message.get("content")
        request_id = message.get("id")

        if not content or not request_id:
            raise ValueError("Invalid message: Missing 'content' or 'id'")

        execute_test_response_and_callback(content, request_id)

        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.info(f"Message processed successfully: {request_id}")

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

port = 5671
rabbitmq_url = f"amqps://{settings.RABBITMQ_USER}:{settings.RABBITMQ_PASS}@{settings.RABBITMQ_URL}:{port}"

def start_worker():
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        ssl_context.check_hostname = True
        ssl_context.load_default_certs()

        params = pika.URLParameters(rabbitmq_url)
        params.ssl_options = pika.SSLOptions(ssl_context)

        connection = pika.BlockingConnection(params)
        channel = connection.channel()

        channel.exchange_declare(exchange="aiProcessingExchange", exchange_type="direct", durable=True)
        channel.queue_declare(queue="aiProcessingQueue", durable=True)
        channel.queue_bind(exchange="aiProcessingExchange", queue="aiProcessingQueue", routing_key="ai.processing.key")

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue="aiProcessingQueue", on_message_callback=process_message)

        logger.info("Worker is waiting for messages...")
        channel.start_consuming()

    except ssl.SSLError as e:
        logger.error(f"SSL 연결 오류 발생: {e}")
    except pika.exceptions.AMQPConnectionError as e:
        logger.error(f"RabbitMQ 연결 오류 발생: {e}")
    except Exception as e:
        logger.error(f"알 수 없는 오류 발생: {e}")
