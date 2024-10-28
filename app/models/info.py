from pydantic import BaseModel
from typing import Optional, Union, List
from datetime import datetime

class DataInfoSTT(BaseModel):
    title: Optional[str]
    ai_stt: Optional[str]

class DataInfoSummary(BaseModel):
    title: Optional[str]
    stance_plaintiff: Optional[str]
    stance_defendant: Optional[str]
    summary_ai: Optional[str]
    judgement: Optional[str]
    fault_rate: Optional[float]

class VoiceInfo(BaseModel):
    status: Optional[str]
    timestamp: Optional[datetime]
    data: Optional[Union[DataInfoSTT]]
