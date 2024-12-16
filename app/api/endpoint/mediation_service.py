import json
import os
import ssl
from fastapi import APIRouter, HTTPException, Header, status, BackgroundTasks
import httpx
from sqlalchemy import text
from datetime import datetime
# from app.db.database import engine
from models.info import DataInfoSummary, VoiceInfo, DataInfoSTT,JudgeRequest,STTRequest
from services.situation_summary import situation_summary_GPT,stt_model,generate_response,test_response
from services.STT import S3SttService
import logging
from core.config import settings
from services.emotion_behavior_situation import RelationshipAnalyzer
import requests
import uuid
import pika
import redis
from app.services.emotion_behavior_situation import RelationshipAnalyzer
router = APIRouter()
logger = logging.getLogger("uvicorn")

analyzer = RelationshipAnalyzer()

@router.post("/speech-to-text", response_model=VoiceInfo, status_code=201)
async def get_voice(request: STTRequest):
    logger.info("get_infos start")
    logger.info(f"audio URL : {request.url}")
    S3SttService.download_file_from_s3(request.url)
    logger.info(f"download_file_from_s3")
    # if not request.url:
    #     raise HTTPException(status_code=400, detail="URL_NOT_PROVIDED")
    #
    # try:
    #     entities = S3SttService.download_file_from_s3(request.url)
    #     logger.info(f"download_file_from_s3")
    #     if "ai_stt" not in entities:
    #         logger.error("STT model did not return expected field: ai_stt")
    #         raise HTTPException(status_code=422, detail="STT_DATA_MISSING")
    # except httpx.HTTPStatusError as e:
    #     logger.error(f"HTTP error during STT processing: {e}")
    #     raise HTTPException(status_code=502, detail="STT_HTTP_ERROR")
    # except httpx.ConnectError as e:
    #     logger.error(f"Connection error during STT processing: {e}")
    #     raise HTTPException(status_code=503, detail="STT_CONNECTION_ERROR")
    # except Exception as e:
    #     logger.error(f"General STT processing error: {e}")
    #     raise HTTPException(status_code=500, detail="STT_PROCESSING_ERROR")

    # response = VoiceInfo(
    #     status="Created",
    #     timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #     data=DataInfoSTT(
    #         script=entities.get("ai_stt")
    #     )
    # )
    # logger.info(f"response : {response}")
    return response

# @router.post("/judgement", response_model=DataInfoSummary, status_code=201)
# async def process_judge(request: JudgeRequest):
#     logger.info(f"Received request data: {request.dict()}")
#     logger.info("Starting judge processing")
#
#     if not request.content:
#         raise HTTPException(status_code=400, detail="CONTENT_NOT_PROVIDED")
#
#     try:
#         # entities = test_response(request.content)
#         entities = await situation_summary_GPT(request.content)
#         # required_fields = ["title", "stance_plaintiff", "stance_defendant", "situation_summary", "judgement", "fault_rate"]
#         required_fields = ["situation_summary", "judgement", "fault_ratios"]
#         # missing_fields = [field for field in required_fields if field not in entities]
#         missing_fields = [field for field in required_fields if field not in entities]
#         if missing_fields:
#             logger.error(f"Missing fields in GPT response: {missing_fields}")
#             raise HTTPException(status_code=422, detail="GPT_DATA_MISSING")
#     except KeyError as e:
#         logger.error(f"Missing field in GPT response: {e}")
#         raise HTTPException(status_code=422, detail="MISSING_GPT_FIELD")
#     except Exception as e:
#         logger.error(f"General error during GPT processing: {e}")
#         raise HTTPException(status_code=500, detail="GPT_PROCESSING_ERROR")
#
#     response = DataInfoSummary(
#         title=entities["situation_summary"]["title"],
#         stancePlaintiff=entities["judgement"]["A_position"],
#         stanceDefendant=entities["judgement"]["B_position"],
#         summaryAi=entities["situation_summary"]["situation_summary"],
#         judgement=entities["judgement"]["conclusion"],
#         faultRate = f"{entities['fault_ratios']['A'] * 100:.2f}"
#         )
#
#     # response = DataInfoSummary(
#     #     title=entities.get("title"),
#     #     stancePlaintiff=entities.get("stance_plaintiff"),
#     #     stanceDefendant=entities.get("stance_defendant"),
#     #     summaryAi=entities.get("situation_summary"),
#     #     judgement=entities.get("judgement"),
#     #     faultRate=entities.get("fault_rate")
#     # )
#     logger.info("Finished judge processing")
#     logger.info(f"판결 응답: {response}")
#     return response
#
#
# # async def read_user_info():
# #     query = text("SELECT user_info FROM user")
# #     async with engine.connect() as conn:
# #         result = await conn.execute(query)
# #         user_info = [row[0] for row in result.fetchall()]
# #     return user_info

################
# @router.post("/judgement", response_model=DataInfoSummary, status_code=201)
# async def process_judge(request: JudgeRequest):
#     logger.info(f"Received request data: {request.dict()}")
#     logger.info("Starting judge processing")
#
#     if not request.content:
#         raise HTTPException(status_code=400, detail="CONTENT_NOT_PROVIDED")
#
#     try:
#         entities = test_response(request.content)
#         required_fields = ["title", "stance_plaintiff", "stance_defendant", "situation_summary", "judgement", "fault_rate"]
#         missing_fields = [field for field in required_fields if field not in entities]
#
#         if missing_fields:
#             logger.error(f"Missing fields in GPT response: {missing_fields}")
#             raise HTTPException(status_code=422, detail="GPT_DATA_MISSING")
#     except KeyError as e:
#         logger.error(f"Missing field in GPT response: {e}")
#         raise HTTPException(status_code=422, detail="MISSING_GPT_FIELD")
#     except Exception as e:
#         logger.error(f"General error during GPT processing: {e}")
#         raise HTTPException(status_code=500, detail="GPT_PROCESSING_ERROR")
#
#     response = DataInfoSummary(
#         title=entities.get("title"),
#         stancePlaintiff=entities.get("stance_plaintiff"),
#         stanceDefendant=entities.get("stance_defendant"),
#         summaryAi=entities.get("situation_summary"),
#         judgement=entities.get("judgement"),
#         faultRate=entities.get("fault_rate")
#     )
#     logger.info("Finished judge processing")
#     logger.info(f"판결 응답: {response}")
#     return response
#########

################
# @router.post("/temptest", status_code=201)
# def testclassify_text(test_request: TestRequest):
#     # 입력 텍스트 받기
#     test_text = test_request.text
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Received test text: {test_text}")
#     print(f"\n입력 텍스트: {test_text}")
#
#     # 모델 분류
#     try:
#         prediction, confidence = classify_text(model, tokenizer, test_text, device, label_map)
#
#         # 결과 출력
#         print(f"예측된 카테고리: {prediction} (확신도: {confidence:.2%})")
#
#         # 응답 생성
#         response = {
#             "prediction": prediction,
#             "confidence": confidence
#         }
#         return response
#
#     except Exception as e:
#         logger.error(f"Error in classify_text: {e}")
#         print(f"오류 발생: {e}")
#         return {"error": "An error occurred while processing the text"}
# #########
#임시 비동기
######
CALLBACK_URL = os.getenv('CALLBACK_URL')
ACCESSTOKEN = os.getenv('ACCESSTOKEN')
# Mock test_response 함수
# def test_response(content: str):
#     """AI 모델 모의 처리"""
#     # 실제 AI 모델 로직을 여기에 추가
#     return {
#         "title": "Generated Title",
#         "stance_plaintiff": "Generated stance for plaintiff",
#         "stance_defendant": "Generated stance for defendant",
#         "situation_summary": "Generated situation summary",
#         "judgement": "Generated judgement text",
#         "fault_rate": 30
#     }

# 백그라운드 작업 함수
def execute_test_response_and_callback(content: str, id:int):
    """
    test_response 호출 후 결과를 CALLBACK_URL로 전송
    """
    try:
        # test_response 실행
        logger.info("Executing test_response function...")
        entities = test_response(content)

        # 필수 필드 검증
        required_fields = [
            "title", "stance_plaintiff", "stance_defendant",
            "situation_summary", "judgement", "fault_rate"
        ]
        missing_fields = [field for field in required_fields if field not in entities]

        if missing_fields:
            logger.error(f"Missing fields in GPT response: {missing_fields}")
            callback_response = {
                "status": False,
                "id": id
            }
        else:
            callback_response = {
                "status": True,
                "accesstoken": ACCESSTOKEN,
                "id": id,
                "title": entities.get("title"),
                "stancePlaintiff": entities.get("stance_plaintiff"),
                "stanceDefendant": entities.get("stance_defendant"),
                "summaryAi": entities.get("situation_summary"),
                "judgement": entities.get("judgement"),
                "faultRate": entities.get("fault_rate")
            }

        # POST 결과를 CALLBACK_URL로 전송
        logger.info(f"Sending POST request to CALLBACK_URL: {CALLBACK_URL}")
        response = requests.post(CALLBACK_URL, json=callback_response)
        logger.info(f"Callback response: {response.status_code}, {response.text}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        # 실패한 경우 콜백 URL로 에러 메시지 전송
        error_response = {"status": "error", "message": str(e)}
        requests.post(CALLBACK_URL, json=error_response)
#####
# 기존 비동기 로직
# @router.post("/judgement", status_code=202)
# async def process_judge(request: JudgeRequest, background_tasks: BackgroundTasks):
#     """
#     요청을 수락하고 202 응답을 반환.
#     BackgroundTasks를 이용해 test_response 호출 후 결과를 CALLBACK_URL로 POST 전송.
#     """
#     logger.info(f"Received request data: {request.dict()}")
#
#     if not request.content:
#         raise HTTPException(status_code=400, detail="CONTENT_NOT_PROVIDED")
#
#     # Background 작업 등록
#     logger.info("Starting background task for judgement processing...")
#     background_tasks.add_task(execute_test_response_and_callback, request.content, request.id)
#
#     # 202 Accepted 응답 반환
#     return {"status": "accepted", "message": "Judgement processing started."}

########
#MQ
def execute_test_response_and_callback(content: str, id: int):
    """
    test_response 호출 후 결과를 CALLBACK_URL로 전송
    """
    try:
        # test_response 실행
        logger.info("Executing test_response function...")
        entities = test_response(content)

        # 필수 필드 검증
        required_fields = [
            "title", "stance_plaintiff", "stance_defendant",
            "situation_summary", "judgement", "fault_rate"
        ]
        missing_fields = [field for field in required_fields if field not in entities]

        if missing_fields:
            logger.error(f"Missing fields in GPT response: {missing_fields}")
            DataInfoSummary = {
                "status": False,
                "id": id
            }
        else:
            DataInfoSummary = {
                "status": True,
                "accesstoken": ACCESSTOKEN,
                "id": id,
                "title": entities.get("title"),
                "stancePlaintiff": entities.get("stance_plaintiff"),
                "stanceDefendant": entities.get("stance_defendant"),
                "summaryAi": entities.get("situation_summary"),
                "judgement": entities.get("judgement"),
                "faultRate": entities.get("fault_rate")
            }

        # POST 결과를 CALLBACK_URL로 전송
        logger.info(f"Sending POST request to CALLBACK_URL: {CALLBACK_URL}")
        response = requests.post(CALLBACK_URL, json=callback_response)
        logger.info(f"Callback response: {response.status_code}, {response.text}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        # 실패한 경우 콜백 URL로 에러 메시지 전송
        error_response = {"status": "error", "message": str(e)}
        requests.post(CALLBACK_URL, json=error_response)

def process_message(ch, method, properties, body):
    """
    RabbitMQ 메시지 소비 후 처리
    """
    try:
        # 메시지 디코딩 및 파싱
        message = json.loads(body)
        content = message.get("content")
        request_id = message.get("id")

        if not content or not request_id:
            raise ValueError("Invalid message: Missing 'content' or 'id'")

        # test_response 실행 및 콜백 전송
        execute_test_response_and_callback(content, request_id)

        # 메시지 처리 완료
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.info(f"Message processed successfully: {request_id}")

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)  # 실패 메시지 재시도 방지
port = 5671
vhost = "/"
rabbitmq_url = f"amqps://{settings.RABBITMQ_URL}:{settings.RABBITMQ_PASS}@{settings.RABBITMQ_URL}:{port}"
def start_worker():
    """
    RabbitMQ 워커 시작
    """
    try:
        # SSL 컨텍스트 설정 (SSL 인증서 검증 비활성화)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = True  # 호스트 이름 검증 비활성화
        # context.verify_mode = ssl.CERT_NONE  # 인증서 검증 비활성화

        # RabbitMQ 연결 설정
        url = rabbitmq_url # 설정에서 RabbitMQ URL 가져오기
        params = pika.URLParameters(url)
        params.ssl_options = pika.SSLOptions(context)

        connection = pika.BlockingConnection(params)  # RabbitMQ 연결
        channel = connection.channel()

        # Exchange 선언
        channel.exchange_declare(exchange="aiProcessingExchange", exchange_type="direct", durable=True)

        # Queue 선언 및 바인딩
        channel.queue_declare(queue="aiProcessingQueue", durable=True)
        channel.queue_bind(exchange="aiProcessingExchange", queue="aiProcessingQueue", routing_key="ai.processing.key")

        # 메시지 소비 시작
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

def process_message(ch, method, properties, body):
    """
    메시지 처리 콜백 함수
    """
    logger.info(f"Received message: {body}")
    # 메시지 처리 로직 추가
    ch.basic_ack(delivery_tag=method.delivery_tag)  # 메시지 확인