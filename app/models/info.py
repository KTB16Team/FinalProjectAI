from pydantic import BaseModel
from typing import Optional, Union, List
from datetime import datetime, date
from services.emotion_behavior_situation import RelationshipAnalyzer, SituationSummary

class DataInfoSTT(BaseModel):
    script: Optional[str]

class DataInfoSummary(BaseModel):
    title: Optional[str]
    stancePlaintiff: Optional[str]
    stanceDefendant: Optional[str]
    summaryAi: Optional[str]
    judgement: Optional[str]
    faultRate: Optional[float]

class VoiceInfo(BaseModel):
    status: Optional[str]
    timestamp: Optional[datetime]
    data: Optional[Union[DataInfoSTT]]

class JudgeRequest(BaseModel):
    content: str 
    nickname: str 
    gender: str 
    birth: date 

class STTRequest(BaseModel):
    url: str

class BehaviorClassificationRequest(BaseModel):
    text: str

class BehaviorClassificationResponse(BaseModel):
    success: bool
    behavior_type: str
    confidence: float
    confidence_level: str