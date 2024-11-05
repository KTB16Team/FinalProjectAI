from pydantic import BaseModel
from typing import Optional, Union, List
from datetime import datetime
from 
class DataInfoSTT(BaseModel):
    title: Optional[str]
    aiStt: Optional[str]

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
