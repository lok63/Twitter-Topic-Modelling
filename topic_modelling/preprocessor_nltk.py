import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import re
try:
    from nltk.tokenize import TweetTokenizer
except ImportError:
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    nltk.download('stopwords')


class NLTKPreprocessor:
    def __init__(self, stopwords, lemmatizer, tokenizer):
        self.stopwords = set(stopwords) if stopwords else set(stopwords)
        self.lemmatizer = lemmatizer
        self.tokenizer = tokenizer


    def remove_digits_and_lower(self, text: str) -> str:
        return " ".join(re.sub(r'\d+', '', text).lower().split())

    def expand_contructions(self, text: str) -> list:
        # https://pypi.org/project/contractions/
        # or
        # https://pypi.org/project/pycontractions/1.0.1/
        return contractions.fix(text)

    def remove_puncs(self, text: str) -> str:
        return re.sub(r'[^\w\s]', '', text)

    # def lemmatize(self, text: str) -> list:
    #     return self.lemmatizer.lemmatize(text)

    def remove_stopwords_and_lematize(self, text: str) -> list:
        return " ".join([self.lemmatizer.lemmatize(w) for w in self.tokenizer.tokenize(text) if w not in self.stopwords])

    def preprocess(self, text: str) -> list:
        text = self.remove_digits_and_lower(text)
        text = self.expand_contructions(text)
        text = self.remove_puncs(text)
        text = self.remove_stopwords_and_lematize(text)
        return text

if __name__ == '__main__':
    nltk_preprocessor = NLTKPreprocessor(
        stopwords=stopwords.words('english'),
        lemmatizer=WordNetLemmatizer(),
        tokenizer=TweetTokenizer()
    )

    text = "I'm 27 and 333I333 am a big fan of #Python, #NLP and #MachineLearning 369"
    print(nltk_preprocessor.preprocess(text))

