from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Optional

class ContentType(str, Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"

class ContentFlag(str, Enum):
    SAFE = "SAFE"
    EXPLICIT = "EXPLICIT"
    SENSITIVE = "SENSITIVE"

class AnalysisResponse(BaseModel):
    is_safe: bool
    confidence_score: float
    flags: List[ContentFlag]
    details: Optional[Dict] = None