from typing import Optional, List

from pydantic import BaseModel

class voice_info(BaseModel):
    situation_stt : Optional[str]

class info(BaseModel):
    situation_summary : Optional[str]
    individual_situation_summary_A : Optional[List[str]]
    individual_situation_summary_B : Optional[str]
    mediate : Optional[str]

