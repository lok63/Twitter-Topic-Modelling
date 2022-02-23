from fastapi import APIRouter
from api_service.api.models.topic_models import TopicModel, TopicModelResponse
from api_service.api.config import get_api_settings
from topic_modelling.pipelines import spacy_pipeline
from typing import List, Optional, Tuple
import pandas as pd
from pathlib import Path
from gensim.test.utils import datapath
from gensim.models import LdaModel

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FROZEN_MODELS_PATH = PROJECT_ROOT / 'frozen_models'
lda = LdaModel.load(datapath(str(FROZEN_MODELS_PATH/'lda_tfidf')))

settings = get_api_settings()
router = APIRouter()

@router.post("/get-topics")
async def inference_topics_from_text(userIn : TopicModel):

    df = pd.DataFrame([{"cleanBody": userIn.text}])
    df = spacy_pipeline.apply(df, column='cleanBody')

    tokenized_text = [df['cleanBody'].values[0]]

    vector = [lda.id2word.doc2bow(text) for text in tokenized_text]
    print(vector)
    topics = sorted(lda[vector][0], key=lambda x: x[1], reverse=True)

    return TopicModelResponse(topics=topics)