from pydantic import BaseModel, Extra, validator

class TopicModel(BaseModel):
    text: str

    class Config:
        extra = Extra.ignore

    @validator('text')
    def check_text(cls, v):
        if v == "":
            raise ValueError("Text cannot be empty")
        return v