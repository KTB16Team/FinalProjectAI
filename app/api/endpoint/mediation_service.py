import json
from fastapi import APIRouter, HTTPException, Header, status
import httpx
from sqlalchemy import text
from datetime import datetime
# from app.db.database import engine
from models.info import DataInfoSummary, VoiceInfo, DataInfoSTT,JudgeRequest,STTRequest, BehaviorClassificationRequest, BehaviorClassificationResponse
from services.situation_summary import situation_summary_GPT,stt_model,generate_response,test_response
import logging
import torch
import os
from transformers import BertTokenizer
from services.emotion_behavior_situation import RelationshipAnalyzer
from services.test_behavior_classification import CustomBERTClassifier, Config
# from app.services.emotion_behavior_situation import RelationshipAnalyzer
router = APIRouter()
logger = logging.getLogger("uvicorn")

analyzer = RelationshipAnalyzer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None

def init_model():
    global model, tokenizer
    try:
        model = CustomBERTClassifier(num_labels=Config.NUM_LABELS)
        model_path = os.path.join("/Users/alice.kim/Desktop/aa/Final/Behavior_classifier.pt")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        print("Model initialized successfully!")
        return True
    except Exception as e:
        print(f"Failed to initialize model: {str(e)}")
        return False

if not init_model():
    raise RuntimeError("Model initialization failed!")

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

@router.post("/judgement", response_model=DataInfoSummary, status_code=201)
async def process_judge(request: JudgeRequest):
    logger.info(f"Received request data: {request.dict()}")
    logger.info("Starting judge processing")
    
    if not request.content:
        raise HTTPException(status_code=400, detail="CONTENT_NOT_PROVIDED")

    try:
        # entities = test_response(request.content)
        entities = await situation_summary_GPT(request.content)
        # required_fields = ["title", "stance_plaintiff", "stance_defendant", "situation_summary", "judgement", "fault_rate"]
        required_fields = ["situation_summary", "judgement", "fault_ratios"]
        # missing_fields = [field for field in required_fields if field not in entities]
        missing_fields = [field for field in required_fields if field not in entities]
        if missing_fields:
            logger.error(f"Missing fields in GPT response: {missing_fields}")
            raise HTTPException(status_code=422, detail="GPT_DATA_MISSING")
    except KeyError as e:
        logger.error(f"Missing field in GPT response: {e}")
        raise HTTPException(status_code=422, detail="MISSING_GPT_FIELD")
    except Exception as e:
        logger.error(f"General error during GPT processing: {e}")
        raise HTTPException(status_code=500, detail="GPT_PROCESSING_ERROR")

    response = DataInfoSummary(
        title=entities["situation_summary"]["title"],
        stancePlaintiff=entities["judgement"]["A_position"],
        stanceDefendant=entities["judgement"]["B_position"],
        summaryAi=entities["situation_summary"]["situation_summary"],
        judgement=entities["judgement"]["conclusion"],
        faultRate = f"{entities['fault_ratios']['A'] * 100:.2f}"
        )
    
    # response = DataInfoSummary(
    #     title=entities.get("title"),
    #     stancePlaintiff=entities.get("stance_plaintiff"),
    #     stanceDefendant=entities.get("stance_defendant"),
    #     summaryAi=entities.get("situation_summary"),
    #     judgement=entities.get("judgement"),
    #     faultRate=entities.get("fault_rate")
    # )
    logger.info("Finished judge processing")
    logger.info(f"판결 응답: {response}")
    return response

@router.post("/classify", response_model=BehaviorClassificationResponse)
def classify_behavior(request: BehaviorClassificationRequest):
    label_map = {
        0: "경쟁형",
        1: "회피형",
        2: "수용형",
        3: "타협형",
        4: "협력형"
    }

    try:
        encoded = tokenizer(
            request.text,
            truncation=True,
            padding='max_length',
            max_length=Config.MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item()

        return BehaviorClassificationResponse(
            success=True,
            behavior_type=label_map[prediction],
            confidence=confidence,
            confidence_level="높음" if confidence >= 0.8 else "중간" if confidence > 0.4 else "낮음"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
    
# async def read_user_info():
#     query = text("SELECT user_info FROM user")
#     async with engine.connect() as conn:
#         result = await conn.execute(query)
#         user_info = [row[0] for row in result.fetchall()]
#     return user_info

# @router.post("/temp_test", response_model=DataInfoSummary, status_code=201)
# def init_model():
#     test_text = "프로젝트 진행 상황이 많이 늦어지고 있어요. 이대로 가다가는 기한 내에 끝내기 힘들 것 같은데, 어떻게 생각하세요?"
#     result = behavior_classification_test(test_text)
#     print("\n테스트 결과:", result)
#     return result