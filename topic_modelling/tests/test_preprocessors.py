from topic_modelling.spacy_preprocessor  import SpacyPreprocessor

def test_spacy_preprocessor():
    spacy_preprocessor = SpacyPreprocessor()
    assert spacy_preprocessor.preprocess_one("Hello world!") == "hello world"