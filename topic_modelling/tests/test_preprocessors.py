from topic_modelling.preprocessor_spacy  import SpacyPreprocessor
from topic_modelling.preprocessor_all import remove_predefined_noise

test_cases = [
    {'in':['amp','https'],
     'out':['https']},
    {'in':['amperage', 'amputate', 'http'],
     'out':['amperage', 'amputate']},

]

def test_remove_predefined_noise():
    for item in test_cases:
        assert (remove_predefined_noise(item['in']) == item['out'])

def test_spacy_preprocessor():
    spacy_preprocessor = SpacyPreprocessor()
    assert spacy_preprocessor.preprocess_one("Hello world!") == "hello world"

if __name__ == '__main__':
    print(remove_predefined_noise(['amp', 'http','https']))