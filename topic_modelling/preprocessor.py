import pandas as pd
import preprocessor as tp
from langdetect import detect, detect_langs
from typing import List, Tuple
import spacy
import os
import re
from utils import timing
import swifter


@timing
def load_data()->pd.DataFrame:
    df1 = pd.read_csv('data/twitter_dataset.csv')
    df2 = pd.read_csv('data/twitter_dataset 2.csv')

    # Make sure the 2 CSV files have the same schema
    print(df1.shape)
    print(df2.shape)

    # Ensure that the 2 DFs have the same shape/columns
    assert list(df1.columns) == list(df2.columns)
    # Ensure that the 2 DFs have the same types in each columns
    assert (df1.dtypes == df2.dtypes).any() == True

    df = pd.concat([df1,df2])
    df['cleanBody'] = df['TweetBody']
    return df

@timing
def clean_hashtags(df:pd.DataFrame) -> pd.DataFrame:
    # Remove hashtags from the text
    df['hashtags'] = df["cleanBody"].apply(lambda x: re.findall(r"#(\w+)", x))
    df['cleanBody'] = df['cleanBody'].apply(lambda x: re.sub(r'#\w+', '', x))
    return df

@timing
def tweet_preprocessor(df:pd.DataFrame) -> pd.DataFrame:
    """
    Removes @mentions, #hashtags. URLs, reserved words (RT, FAV), emojis and Smiley faces
    https://github.com/s/preprocessor
    """
    tp.set_options(tp.OPT.URL,
                   tp.OPT.MENTION,
                   tp.OPT.HASHTAG,
                   tp.OPT.RESERVED,
                   tp.OPT.SMILEY,
                   tp.OPT.EMOJI,
                   # tp.OPT.NUMBER,
                   )
    df['cleanBody'] = df["cleanBody"].swifter.apply(lambda x: tp.clean(x))
    return df

@timing
def spacy_preprocessor(df:pd.DataFrame) -> pd.DataFrame:
    """
    Tokenize, lowercase, remove stop words, lemmatize, remove punctuation
    """
    nlp = spacy.load('en_core_web_sm')
    df['cleanBody'] = df["cleanBody"].apply(lambda x: " ".join([token.lemma_ for token in nlp(x) if not token.is_stop]))
    return df

@timing
def lang_detector(df:pd.DataFrame) -> pd.DataFrame:
    """
    Detect the language of each tweet and add the language to the dataframe
    https://github.com/Mimino666/langdetect
    """
    df['lang'] = df['cleanBody'].swifter.apply(lambda x: get_language(x))
    return df

def get_language(text: str) -> Tuple[str, float]:
    """
     Detect the language of a given text and return the language and the probability. The langdetect API is able to detect multible langyes. but we choose to
     return the first language.When the classifier fails to detect the language, we want to default to UNK.
    """
    try:
        return detect(text)
    except:
        return 'UNK', -1

@timing
def spacy_lang_detector(df:pd.DataFrame) -> pd.DataFrame:
    import spacy
    from spacy.language import Language
    from spacy_langdetect import LanguageDetector

    def get_lang_detector(nlp, name):
        return LanguageDetector()

    # nlp = spacy.load("en_core_web_lg", disable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer", "textcat", "ner"])
    nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer","textcat", "ner"])
    Language.factory("language_detector", func=get_lang_detector)
    nlp.add_pipe('language_detector', last=True)

    text = 'This is an english text.'
    doc = nlp(text)
    print(doc._.language)
    # df['lang'] = df['cleanBody'].swifter.apply(lambda x: get_lang_spacy(x, nlp))
    print(nlp.pipe_names)

    docs = df['cleanBody'].tolist()
    res = [(doc._.language['language'], doc._.language['score']) for doc in nlp.pipe(docs, batch_size=100, n_process=-1)]
    langs , probs = zip(*res)
    df['lang'] = list(langs)
    df['lang_probs'] = list(probs)


    # df['lang'] = [doc._.language['language'] for doc in nlp.pipe(docs,
    #                                                              # disable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer","textcat", "ner"],
    #                                                              batch_size=100,
    #                                                              n_process=-1)]
    print(nlp.pipe_names)
    return df

def get_lang_spacy(text: str, nlp) -> Tuple[str, float]:
    """
    Detect the language of a given text and return the language and the probability. The langdetect API is able to detect multible langyes. but we choose to
    return the first language.When the classifier fails to detect the language, we want to default to UNK.
    """


    try:
        doc = nlp(text)
        return doc._.language['language'], doc._.language['score']
    except:
        return 'UNK', -1



df = load_data()
df = df.reset_index(drop=True)
df = df.iloc[:1000]
# df_no_rt = df.loc[df["TweetRetweetFlag"] == False]

df_clean = tweet_preprocessor(df)
# df_clean = lang_detector(df_clean)
df_clean = spacy_lang_detector(df_clean)
