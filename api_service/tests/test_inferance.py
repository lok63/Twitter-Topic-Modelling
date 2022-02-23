from fastapi.testclient import TestClient
from api_service.main import app
import numpy as np

client = TestClient(app)


def test_inference_api_with_empty_text():
    # When text is empty
    text= ""
    # Then the response should raise an error
    response = client.post("get-topics", json={'text': text})
    assert response.status_code == 422
    assert response.json()['detail'][0]['type'] == "value_error"
    assert response.json()['detail'][0]['msg'] == "Text cannot be empty"

def test_inference_api_with_normal_text():
    # When text is:
    text= "This is a lovely and amazing city. I love Paris"
    response = client.post("get-topics", json={'text': text})
    # Then the response should return some number of topics
    assert response.status_code == 200
    topics = response.json()['topics']
    assert len(topics) > 0
    assert type(topics) == list
    assert type(topics[0][0]) == int
    assert type(topics[0][1]) == float


def test_inference_api_with_oov_words():
    # When text has only words that are not present in the corpus
    text= "Αν το διαβάζεις αυτό, είσαι μάγκας"
    response = client.post("get-topics", json={'text': text})
    # Then the response should return topics but the probability of each topic should be roughly the same
    assert response.status_code == 200
    topics = response.json()['topics']
    topics_prob = [np.round(topic[1], decimals=2) for topic in topics]
    assert all(element == topics_prob[0] for element in topics_prob)