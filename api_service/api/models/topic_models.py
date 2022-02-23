from pydantic import BaseModel, Extra, validator
from typing import Tuple, List, Dict
import numpy as np

class TopicModel(BaseModel):
    text: str

    class Config:
        extra = Extra.ignore

    @validator('text')
    def check_text(cls, v):
        if v == "":
            raise ValueError("Text cannot be empty")
        return v


class TopicModelResponse(BaseModel):
    topics: List[Tuple[int, float]]
