from pydantic import BaseModel
from typing import Optional, Union, List
from datetime import datetime
from app.services.emotion_behavior_situation import RelationshipAnalyzer, SituationSummary

class DataInfoSTT(BaseModel):
    title: Optional[str]
    script: Optional[str]

class DataInfoSummary(BaseModel):
    title: Optional[str]
    stancePlaintiff: Optional[str]
    stanceDefendant: Optional[str]
    summaryAi: SituationSummary
    judgement: 
    faultRate: 

class VoiceInfo(BaseModel):
    status: Optional[str]
    timestamp: Optional[datetime]
    data: Optional[Union[DataInfoSTT]]
