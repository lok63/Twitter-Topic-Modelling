from topic_modelling.preprocessor_tweet import tweet_preprocessor as tp
from topic_modelling.preprocessor_all import tweet_preprocessor, spacy_preprocessor, load_data
from topic_modelling.preprocessor_spacy import SpacyPreprocessor
from topic_modelling.preprocessor_nltk import NLTKPreprocessor
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from gensim.utils import tokenize
from gensim.models.phrases import Phrases
from utils import timing
from typing import List,Callable
import pandas as pd

spp = SpacyPreprocessor(model='en_core_web_lg')

nktk_pp = NLTKPreprocessor(
        stopwords=stopwords.words('english'),
        lemmatizer=WordNetLemmatizer(),
        tokenizer=TweetTokenizer()
    )

class Pipeline:
    def __init__(self, transformers: List[Callable]):
        self.transformers = transformers

    def apply(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        for transformer in self.transformers:
            df = transformer(df, column=column)

        return df

@timing
def tweet_preprocessor(df:pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].swifter.apply(lambda x: tp(x))
    return df

@timing
def nltk_preprocessor(df:pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].swifter.apply(lambda x: nktk_pp.preprocess(x))
    return df

@timing
def spacy_preprocessor(df:pd.DataFrame, column: str) -> pd.DataFrame:
    print(":: Spacy preprocessor -> cleaning, this might take 1-2 minutes....")
    df[column] = spp.preprocess_batch(df[column])
    return df

@timing
def drop_empty(df:pd.DataFrame, column: str) -> pd.DataFrame:
    return df.loc[df[column] != '']

@timing
def tokenizer_transformer(df:pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].swifter.apply(lambda x: list(tokenize(x)))
    return df

@timing
def ngrammer_2_3(df:pd.DataFrame, column: str) -> pd.DataFrame:
    bigram_model = Phrases(df[column], min_count=5, threshold=10)
    df[column] = df[column].swifter.apply(lambda x: bigram_model[x])
    return df

@timing
def reset_index(df:pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    return df


basic_pipeline = Pipeline(
    transformers=[
        reset_index,
        tweet_preprocessor,
        nltk_preprocessor,
        drop_empty,
        reset_index,
        tokenizer_transformer,
        ngrammer_2_3,
    ]
)

spacy_pipeline = Pipeline(
    transformers=[
        reset_index,
        tweet_preprocessor,
        spacy_preprocessor,
        drop_empty,
        reset_index,
        tokenizer_transformer,
        ngrammer_2_3,
    ]
)

if __name__ == '__main__':
    df = load_data()
    df = basic_pipeline.apply(df, column='cleanBody')
