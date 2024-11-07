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
    faultRate: Optional[Union[float, int, str]]

class VoiceInfo(BaseModel):
    status: Optional[str]
    timestamp: Optional[datetime]
    data: Optional[Union[DataInfoSTT]]

class JudgeRequest(BaseModel):
    content: str = Field(..., description="대화록이 비었습니다.")
    nickname: str = Field(..., description="유저명이 비었습니다.")
    gender: str = Field(..., description="성별이 비었습니다.")
    birth: date = Field(..., description="생년월일이 비었습니다.")

class STTRequest(BaseModel):
    url: str