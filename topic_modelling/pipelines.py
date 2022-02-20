from topic_modelling.preprocessor_tweet import tweet_preprocessor as tp
from topic_modelling.preprocessor_all import tweet_preprocessor, spacy_preprocessor, load_data
from topic_modelling.preprocessor_spacy import SpacyPreprocessor
from topic_modelling.preprocessor_nltk import NLTKPreprocessor
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from gensim.utils import tokenize
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
def reset_index(df:pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    return df


basic_pipeline = Pipeline(
    transformers=[
        # spacy_preprocessor,
        nltk_preprocessor,
        tweet_preprocessor,
        drop_empty,
        reset_index,
        tokenizer_transformer,
    ]
)

spacy_pipeline = Pipeline(
    transformers=[
        tweet_preprocessor,
        spacy_preprocessor,
        drop_empty,
        reset_index,
        tokenizer_transformer,
    ]
)

if __name__ == '__main__':
    df = load_data()
    df = basic_pipeline.apply(df, column='cleanBody')
