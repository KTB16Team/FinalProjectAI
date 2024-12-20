from pydantic import BaseModel
from typing import Optional, Union, List
from datetime import datetime, date
from services.emotion_behavior_situation import RelationshipAnalyzer, SituationSummary

class DataInfoOCR(BaseModel):
    status: bool
    url: str
    script: Optional[str]
    accessKey: str

class DataInfoSTT(BaseModel):
    script: Optional[str]

class DataInfoSummary(BaseModel):
    status: bool
    id: int
    title: Optional[str]
    stancePlaintiff: Optional[str]
    stanceDefendant: Optional[str]
    summaryAi: Optional[str]
    judgement: Optional[str]
    faultRate: Optional[float]
    accessKey: str

class VoiceInfo(BaseModel):
    status: Optional[str]
    timestamp: Optional[datetime]
    data: Optional[Union[DataInfoSTT]]

class JudgeRequest(BaseModel):
    id: int
    content: str
    nickname: str 
    gender: str 
    birth: date 

class STTRequest(BaseModel):
    url: str


class ConflictAnalysisRequest(BaseModel):
    content: str
    nickname: Optional[str] = None
    gender: Optional[str] = None
    birth: Optional[str] = None
    id: Optional[str] = "id"


class ConflictAnalysisResponseData(BaseModel):
    id: str
    title: str
    stancePlaintiff: str
    stanceDefendant: str
    summaryAi: str
    judgement: str
    faultRate: float

class ConflictAnalysisResponse(BaseModel):
    status: str
    method: str
    data: ConflictAnalysisResponseData

# class BehaviorClassificationRequest(BaseModel):
#     text: str

class BehaviorClassificationResponse(BaseModel):
    success: bool
    behavior_type: str
    confidence: float
    confidence_level: str


# class EmpathyRequest(BaseModel):
#     utterances: List[dict]

# class EmpathyResponse(BaseModel):
#     success: bool
#     empathy_scores: List[float]
#     confidence_level: str

# class EmotionAnalysisRequest(BaseModel):
#     conversation: List[str]  # 대화 내용, 이 필드를 사용해 감정 분석

# class EmotionAnalysisResponse(BaseModel):
#     success: bool
#     results: List[dict]
