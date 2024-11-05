from pydantic import BaseModel
from typing import Optional, Union, List
from datetime import datetime

class DataInfoSTT(BaseModel):
    title: Optional[str]
    aiStt: Optional[str]

class DataInfoSummary(BaseModel):
    title: Optional[str]
    stancePlaintiff: Optional[str]
<<<<<<< Updated upstream
    stance_defendant: Optional[str]
    summary_ai: Optional[str]
=======
    stanceDefendant: Optional[str]
    summaryAi: Optional[str]
>>>>>>> Stashed changes
    judgement: Optional[str]
    faultRate: Optional[float]

class VoiceInfo(BaseModel):
    status: Optional[str]
    timestamp: Optional[datetime]
    data: Optional[Union[DataInfoSTT]]
