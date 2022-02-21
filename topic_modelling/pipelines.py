from topic_modelling.preprocessor_tweet import tweet_preprocessor as tp
from topic_modelling.preprocessor_tweet import demoji_from_text
from topic_modelling.preprocessor_all import load_data, get_language, remove_predefined_noise
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
def demoji_preprocessor(df:pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].swifter.apply(lambda x: demoji_from_text(x))
    return df

@timing
def nltk_preprocessor(df:pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].swifter.apply(lambda x: nktk_pp.preprocess(x))
    return df

@timing
def spacy_preprocessor(df:pd.DataFrame, column: str) -> pd.DataFrame:
    print(":: Spacy preprocessor -> cleaning, this might take 1-2 minutes....")
    df[column] = df[column].swifter.apply(lambda x: spp.preprocess_one(x))
    # df[column] = spp.preprocess_batch(df[column])
    return df

@timing
def lang_detector(df:pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Detect the language of each tweet and add the language to the dataframe
    https://github.com/Mimino666/langdetect
    """
    df['lang'] = df[column].swifter.apply(lambda x: get_language(x))
    return df

@timing
def lang_detect_spacy(df:pd.DataFrame, column: str) -> pd.DataFrame:
    print(":: Detecting language using Spacy -> this might take 1-2 minutes....")
    df[['lang', 'prob']] = df[column].apply(spp.detect_language)
    return df

@timing
def drop_empty(df:pd.DataFrame, column: str) -> pd.DataFrame:
    return df.loc[df[column] != '']

@timing
def tokenizer_transformer(df:pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].swifter.apply(lambda x: list(tokenize(x)))
    return df

@timing
def predefined_denoiser(df:pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].swifter.apply(lambda x: remove_predefined_noise(x))
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
        demoji_preprocessor,
        tweet_preprocessor,
        nltk_preprocessor,
        drop_empty,
        reset_index,
        tokenizer_transformer,
        predefined_denoiser,
        ngrammer_2_3,
    ]
)

spacy_pipeline = Pipeline(
    transformers=[
        reset_index,
        demoji_preprocessor,
        tweet_preprocessor,
        spacy_preprocessor,
        drop_empty,
        reset_index,
        tokenizer_transformer,
        predefined_denoiser,
        ngrammer_2_3,
    ]
)

analytics_pipeline = Pipeline(
    transformers=[
        reset_index,
        demoji_preprocessor,
        tweet_preprocessor,
        nltk_preprocessor,
        # drop_empty,
        reset_index,
        lang_detector,
        tokenizer_transformer,
        predefined_denoiser,
        ngrammer_2_3,
    ]
)
if __name__ == '__main__':
    df = load_data()
    # df = analytics_pipeline.apply(df, column='cleanBody')
    df = lang_detector(df, column='cleanBody')

    # spp.detect_language(['this is an english'], batch_size=1, n_process=1)
