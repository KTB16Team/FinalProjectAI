import json
import os
from fastapi import APIRouter, HTTPException, Header, status, BackgroundTasks
import httpx
from sqlalchemy import text
from datetime import datetime
# from app.db.database import engine
from models.info import DataInfoSummary, VoiceInfo, DataInfoSTT,JudgeRequest,STTRequest
from services.situation_summary import situation_summary_GPT,stt_model,generate_response,test_response
# from services.behavior_classification import
import logging
from services.emotion_behavior_situation import RelationshipAnalyzer
import requests
# import uuid
# import pika
# import redis
# from app.services.emotion_behavior_situation import RelationshipAnalyzer
router = APIRouter()
logger = logging.getLogger("uvicorn")

analyzer = RelationshipAnalyzer()

@router.post("/speech-to-text", response_model=VoiceInfo, status_code=201)
async def get_voice(request: STTRequest):
    logger.info("get_infos start")
    logger.info(f"audio URL : {request.url}")

    if not request.url:
        raise HTTPException(status_code=400, detail="URL_NOT_PROVIDED")

    try:
        entities = stt_model(request.url)
        if "ai_stt" not in entities:
            logger.error("STT model did not return expected field: ai_stt")
            raise HTTPException(status_code=422, detail="STT_DATA_MISSING")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during STT processing: {e}")
        raise HTTPException(status_code=502, detail="STT_HTTP_ERROR")
    except httpx.ConnectError as e:
        logger.error(f"Connection error during STT processing: {e}")
        raise HTTPException(status_code=503, detail="STT_CONNECTION_ERROR")
    except Exception as e:
        logger.error(f"General STT processing error: {e}")
        raise HTTPException(status_code=500, detail="STT_PROCESSING_ERROR")

    response = VoiceInfo(
        status="Created",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data=DataInfoSTT(
            script=entities.get("ai_stt")
        )
    )
    logger.info(f"response : {response}")
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

@router.post("/judgement", status_code=202)
async def process_judge(request: JudgeRequest, background_tasks: BackgroundTasks):
    """
    요청을 수락하고 202 응답을 반환.
    BackgroundTasks를 이용해 test_response 호출 후 결과를 CALLBACK_URL로 POST 전송.
    """
    logger.info(f"Received request data: {request.dict()}")

    if not request.content:
        raise HTTPException(status_code=400, detail="CONTENT_NOT_PROVIDED")

    # Background 작업 등록
    logger.info("Starting background task for judgement processing...")
    background_tasks.add_task(execute_test_response_and_callback, request.content, request.id)

    # 202 Accepted 응답 반환
    return {"status": "accepted", "message": "Judgement processing started."}

