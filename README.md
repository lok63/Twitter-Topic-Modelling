# Twitter-Topic-Modelling
This is a solution using Gensim, a Twitter sample and FastAPI to train a topic model and perform topic analysis on Jupyter Notebooks.
The solution structure is as follows:

## 1. Project Files and Structure
### topic_modelling
  * Multiple Preprocessing functions
  * Models (Abstract classes with different implementations for Topic Modelling)
  * Pipelines (A set of automated preprocessing rules to be used for Topic Analysis and Model Training)
  * Tests (Unit tests for pre-processing functions)

### notebooks
Here we can use the pipelines defined in `topic_modelling` to explore our Dataset and train our models.

#### EDA on Tweets
* Examine our dataset
* pre-process data and extract n-grams
  * find trending n-grams and hashtags
  * Examine tweets by language
#### Topic Modelling
* Train a baseline LDA Model
  * Hyperparameter tuning and find the best number of models
* Train an LDA with Bow and TfIDF
* Topic Visualisation and analysis
* Model Selection
* Analysis and insights from topics

## frozen_models
This is a collection of models trained on the dataset.

## api_service
A Fast API service to serve the topic models and perform topic analysis on an endpoint and a new user Tweet
The solution automatically loads the pre-trained LDA-TFIDF and the pre-processing pipelines and uses an endpoint to perform topic analysis on a new user Tweet.
Unit tests have been provided to test the API service.


## 2. Getting Started
For this solution make sure you have Python 3.8 installed on your machine and have a new virtual environment activated.

We run everything from the root directory of the project. So make sure you are in the root directory of the project and export the `PYTHONPATH` variable.
```bash
# cd to root directory of project
cd Twitter-Topic-Modelling
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

If you have conda run the following command:
```bash
conda create -n tweet python=3.8
conda activate tweet
```

If you have `pyenv` installed you can activate the environment with the following command:
```bash
pyenv virtualenv 3.8.12 tweet
pyenv activate tweet
````

Install the requirements
```bash
pip install -r requirements.txt
```
### 2.1 Notebooks
Run the notebooks to explore the dataset and train the models.
```bash
jupyter notebook
```

### 2.2 API
The API was designed to be used with a REST API and deployed on a server as a microservice. For that reason, I created it's own requirements.txt file and also requires a`.env` file to be created.
In order to run it, we should create a `.env` file with the following variables:
```bash
PROJECT_ID=cwt
PROJECT_ENV=development
PORT=5001
```

#### Run API
```bash
# from the root directory of the project
python api_service/main.py  
```

You can play around with the API by sending a POST request to the following endpoint: http://0.0.0.0:5001/get-topics
or
by launching the Swagger API on this link http://0.0.0.0:5001/docs
![Swagger Image](https://github.com/lok63/Twitter-Topic-Modelling/swagger.png?raw=true)


### Run the API Tests
```bash
# from the root directory of the project
pytest api_service/tests  
```

