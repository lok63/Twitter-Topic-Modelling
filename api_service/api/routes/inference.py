from fastapi import APIRouter
from api.models.topic_models import TopicModel
from api.config import get_api_settings
from topic_modelling.pipelines import spacy_pipeline
from typing import List, Optional
import pandas as pd

settings = get_api_settings()
router = APIRouter()

@router.post("/get-topics")
async def inference_topics_from_text(userIn : TopicModel):
    #TODO: load model and get topics
    # Preprocess text

    df = pd.DataFrame([{"cleanBody": userIn.text}])
    df = spacy_pipeline.apply(df, column='cleanBody')
    pre_processed_text = df['cleanBody'].values[0]

    return pre_processed_text