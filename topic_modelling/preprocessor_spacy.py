from spacy_langdetect import LanguageDetector
from spacy.language import Language
from typing import List, Union, Tuple
from utils import timing
import spacy

class SpacyPreprocessor:
    def __init__(self, model: str = 'en_core_web_sm'):
        self.nlp = spacy.load(model, disable=["tok2vec", "textcat", "ner"])
        self.language_model = self._init_language_model(model)


    def _init_language_model(self, model: str = 'en_core_web_sm'):
        nlp = spacy.load(model, disable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer", "textcat", "ner"])
        Language.factory("language_detector", func=self._get_lang_detector)
        nlp.add_pipe('language_detector', last=True)
        return nlp

    def _get_lang_detector(self, nlp, name):
        return LanguageDetector()

    def preprocess_one(self, doc: Union[str,spacy.tokens.doc.Doc]) -> str:
        """
        Tokenize, lowercase, remove stop words, lemmatize, remove punctuation
        """
        try:
            if isinstance(doc, str):
                doc = self.nlp(doc)
            return " ".join([str(token.lemma_).lower() for token in doc if token.is_alpha and token.text.lower() and not token.is_stop])
        # When doc is '' it will throw a valueError
        except ValueError:
            return ""
        except Exception as e:
            raise e

    # @timing
    def preprocess_batch(self, docs: List[str], batch_size: int = 1000) -> List[str]:
        clean_docs = []
        for doc in self.nlp.pipe(docs, batch_size=batch_size, n_process=-1):
            clean_docs.append(self.preprocess_one(doc))
        return clean_docs

    # @timing
    def detect_language(self, docs: List[str], batch_size: int = 1000, n_process=-1) -> Tuple[List[str], List[float]]:

        langs, probs = zip(*[(doc._.language['language'], doc._.language['score']) for doc in
                             self.language_model.pipe(docs, batch_size=batch_size, n_process=n_process)])
        return langs, probs