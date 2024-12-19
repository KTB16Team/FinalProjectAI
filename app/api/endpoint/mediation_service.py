import json
from fastapi import APIRouter, HTTPException, Header, status
import httpx
from sqlalchemy import text
from datetime import datetime
# from app.db.database import engine
from models.info import DataInfoSummary, VoiceInfo, DataInfoSTT,JudgeRequest,STTRequest, ConflictAnalysisRequest,ConflictAnalysisResponseData, ConflictAnalysisResponse
from services.situation_summary import situation_summary_GPT,stt_model,generate_response,test_response
import logging
import torch
import os
import numpy as np
from services.score_multi import ConflictAnalyzer
from transformers import BertTokenizer, AutoTokenizer, AutoModel
from services.empathy_score import DialogueEmpathyModel
from services.emotion_behavior_situation import RelationshipAnalyzer
from services.BERTbasedcontext import EmotionAnalyzer
from services.test_behavior_classification import CustomBERTClassifier, Config
# from app.services.emotion_behavior_situation import RelationshipAnalyzer
router = APIRouter()
logger = logging.getLogger("uvicorn")

analyzer = RelationshipAnalyzer()
conflict_analyzer = ConflictAnalyzer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = None
# tokenizer = None
# emotion_analyzer = None  # 감정 분석 모델
# empathy_model = None  # 공감 예측 모델
# empathy_tokenizer = None  # 공감 예측 토크나이저
# emotion_model = None  # 감정 분석 모델
# emotion_tokenizer = None  # 감정 분석 토크나이저
# bert_model = None

# def init_model():
#     global model, tokenizer
#     try:
#         model = CustomBERTClassifier(num_labels=Config.NUM_LABELS)
#         model_path = os.path.join("/Users/alice.kim/Desktop/aa/Final/Behavior_classifier.pt")
#         state_dict = torch.load(model_path, map_location=device)
#         model.load_state_dict(state_dict)
#         model.to(device)
#         model.eval()

#         tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
#         print("Model initialized successfully!")
#         return True
#     except Exception as e:
#         print(f"Failed to initialize model: {str(e)}")
#         return False

# def init_empathy_model():
#     global empathy_model, empathy_tokenizer, bert_model
#     try:
#         # Empathy Model Initialization
#         bert_model = AutoModel.from_pretrained('skt/kobert-base-v1').to(device)
#         bert_model.eval() 
#         empathy_model = DialogueEmpathyModel(
#             input_size=768,  # BERT hidden size
#             hidden_size=256,
#             num_speakers=10  # 화자 수는 상황에 맞게 설정
#         )
#         empathy_model_path = "/Users/alice.kim/Desktop/aa/Final/best_model.pt" 
#         state_dict = torch.load(empathy_model_path, map_location=device)
#         empathy_model.load_state_dict(state_dict)
#         empathy_model.to(device)
#         empathy_model.eval()

#         empathy_tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
#         print("Empathy Model initialized successfully!")
#         return True
#     except Exception as e:
#         print(f"Failed to initialize empathy model: {str(e)}")
#         return False

# def init_emotion_analyzer():
#     global emotion_analyzer
#     try:
#         # Explicitly set device
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Initialize emotion analyzer
#         emotion_analyzer = EmotionAnalyzer()
        
#         # Load model with proper device mapping
#         model_path = os.path.join("/Users/alice.kim/Desktop/aa/Final/BERTbasedemotion_model.pt")
#         state_dict = torch.load(model_path, map_location=device)
        
#         # Load state dict and ensure model is on correct device
#         emotion_analyzer.model.load_state_dict(state_dict)
#         emotion_analyzer.model = emotion_analyzer.model.to(device)
#         emotion_analyzer.model.eval()
        
#         print(f"Emotion Analyzer Model initialized successfully on {device}!")
#         return True
        
#     except Exception as e:
#         print(f"Failed to initialize emotion analyzer model: {str(e)}")
#         return False

# if not init_model():
#     raise RuntimeError("Model initialization failed!")
# if not init_emotion_analyzer():
#     raise RuntimeError("Emotion model initialization failed!")
# if not init_empathy_model():
#     raise RuntimeError("Empathy model initialization failed!")
@router.post("/analyze-conflict", response_model=ConflictAnalysisResponseData, status_code=201)
async def analyze_conflict(request: ConflictAnalysisRequest):
    logger.info(f"Received conflict analysis request: {request.dict()}")

    if not request.content:
        logger.error("CONTENT_NOT_PROVIDED")
        raise HTTPException(status_code=400, detail="CONTENT_NOT_PROVIDED")
    
    try:
        # ConflictAnalyzer를 사용하여 내용 분석
        analysis_result = await conflict_analyzer.analyze_content(
            content=request.content,
            request_id=request.id
        )
        
        if analysis_result["status"] != "success":
            logger.error(f"Conflict analysis failed: {analysis_result.get('message', 'Unknown error')}")
            raise HTTPException(status_code=500, detail="Conflict analysis failed")
        
        # 백엔드 서버로 전송할 데이터 준비
        backend_payload = analysis_result['data']
        
        # 백엔드 서버 URL 설정 -> 이부분 수정해야 함
        backend_server_url = os.getenv("BACKEND_SERVER_URL", "https://api.ktb-aimo.link//api/v1/private-posts/judgement/callback")
        
        # 백엔드 서버로 데이터 전송
        async with httpx.AsyncClient() as client:
            backend_response = await client.post(backend_server_url, json=backend_payload)
            
            if backend_response.status_code != 200:
                logger.error(f"Failed to send data to backend server: {backend_response.status_code}")
                raise HTTPException(status_code=502, detail="Failed to send data to backend server")
        
        # 응답 생성
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

# @router.post("/classify", response_model=BehaviorClassificationResponse)
# def classify_behavior(request: BehaviorClassificationRequest):
#     label_map = {
#         0: "경쟁형",
#         1: "회피형",
#         2: "수용형",
#         3: "타협형",
#         4: "협력형"
#     }

#     try:
#         encoded = tokenizer(
#             request.text,
#             truncation=True,
#             padding='max_length',
#             max_length=Config.MAX_LENGTH,
#             return_tensors="pt"
#         )

#         input_ids = encoded['input_ids'].to(device)
#         attention_mask = encoded['attention_mask'].to(device)

#         with torch.no_grad():
#             logits, _ = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask
#             )
#             probabilities = torch.softmax(logits, dim=1)
#             prediction = torch.argmax(logits, dim=1).item()
#             confidence = probabilities[0][prediction].item()

#         return BehaviorClassificationResponse(
#             success=True,
#             behavior_type=label_map[prediction],
#             confidence=confidence,
#             confidence_level="높음" if confidence >= 0.8 else "중간" if confidence > 0.4 else "낮음"
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# @router.post("/emotion-analysis", response_model=EmotionAnalysisResponse)
# async def predict_emotion(request: EmotionAnalysisRequest):
#     try:
#         conversation = request.conversation
#         logger.info(f"Received conversation for analysis: {conversation}")
        
#         if emotion_analyzer.model is None:
#             logger.error("Model not properly initialized")
#             raise HTTPException(status_code=500, detail="Model not initialized")
            
#         # 모델이 eval 모드인지 확인
#         emotion_analyzer.model.eval()
        
#         # 예측값 로깅 추가
#         results = emotion_analyzer.analyze_conversation(conversation)
#         logger.info(f"Raw prediction results: {results}")
        
#         scores = [result['emotion_score'] for result in results]
#         logger.info(f"Emotion scores distribution - min: {min(scores):.3f}, max: {max(scores):.3f}, mean: {sum(scores)/len(scores):.3f}")
        
#         return EmotionAnalysisResponse(success=True, results=results)

#     except Exception as e:
#         logger.error(f"Emotion analysis failed with error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Emotion analysis failed: {str(e)}")
    
# @router.post("/empathy", response_model=EmpathyResponse)

# async def predict_empathy(request: EmpathyRequest):
#     try:
#         dialogue_texts = [utterance['text'] for utterance in request.utterances]
#         batch_size = len(dialogue_texts)
#         sequence_length = len(dialogue_texts)
        
#         # speaker_ids를 [batch_size]로 만듦
#         speaker_ids = [utterance['speaker_id'] for utterance in request.utterances]
#         speaker_ids = torch.tensor([speaker_ids] * sequence_length, device=device)
#         # BERT 인코딩
#         inputs = empathy_tokenizer(
#             dialogue_texts, 
#             padding=True, 
#             truncation=True, 
#             max_length=128, 
#             return_tensors="pt"
#         ).to(device)

#         with torch.no_grad():
#             # BERT 출력 얻기
#             bert_outputs = bert_model(
#                 input_ids=inputs['input_ids'],
#                 attention_mask=inputs['attention_mask']
#             )
            
#             # last_hidden_state 전체를 사용 [batch_size, seq_len, hidden_size]
#             utterance_embeddings = bert_outputs.last_hidden_state
            
#             # attention mask를 batch_size 길이의 1D 텐서로
#             attention_mask = inputs['attention_mask'].bool()
            
#             logger.info(f"Batch size: {batch_size}")
#             logger.info(f"Initial speaker_ids shape: {speaker_ids.shape}")
#             logger.info(f"Utterance embeddings shape: {utterance_embeddings.shape}")
#             logger.info(f"Attention mask shape: {attention_mask.shape}")

#             # 모델 예측
#             empathy_outputs = empathy_model(
#                 utterances=utterance_embeddings,
#                 speaker_ids=speaker_ids,
#                 attention_mask=attention_mask
#             )
            
#             # 출력 처리
#             empathy_scores = empathy_outputs.squeeze().cpu().numpy()
#             if isinstance(empathy_scores, float):
#                 empathy_scores = [empathy_scores]
#             elif len(empathy_scores.shape) == 0:
#                 empathy_scores = [float(empathy_scores)]
#             else:
#                 empathy_scores = empathy_scores.tolist()

#         # 결과 처리
#         avg_score = sum(empathy_scores) / len(empathy_scores)
#         confidence_level = (
#             "낮음" if avg_score <= 0.39 else
#             "중간" if avg_score <= 0.79 else
#             "높음"
#         )

#         logger.info(f"Final empathy scores: {empathy_scores}")
#         logger.info(f"Confidence level: {confidence_level}")

#         return EmpathyResponse(
#             success=True,
#             empathy_scores=empathy_scores,
#             confidence_level=confidence_level
#         )

#     except Exception as e:
#         logger.error(f"Empathy prediction failed: {str(e)}")
#         logger.error(f"Full error details: {e.__class__.__name__}: {str(e)}")
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Empathy prediction failed: {str(e)}"
#         )

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