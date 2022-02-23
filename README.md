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
![Swagger Image](https://github.com/lok63/Twitter-Topic-Modelling/blob/master/swagger.png)


### Run the API Tests
```bash
# from the root directory of the project
pytest api_service/tests  
```


## 3. Future work and improvements

### 3.1 EDA
* Identify popular users based on their followers and likes count. From there, analyze their tweets
* Language detection doesn't perform that well. We still have some False Positives. Try to fine tune the lang classifier on tweets

### 3.2 Pre-processors
* Try to initialise the n-grams using the words used from Google Vectors. Since this corpus was created in 2016, enrich it with more data


* Spacy's rules is to only retrieve alphabetical tokens. So there are some languages with different characters such as Greek or Japanase that will get excluded. 
* Another issue is that we are loading the english stopwords. So a lot of stopwords from other languages are passing through the pre-processor.
  * If we manage to improve the lang detection, we can dynamically load the specific language model from spacy and pre-process based on the language and it's own stopwords
* Use a NER and store all entities and the labels in a different column

### 3.3 Models
* Identify how to use HierarchicalModel to get the best number of topics
* Expirement with the Word2Vec model
* Create another Visualisation using the Ensemble model
* Export the whole object of the model and it's built in functions instead of the binary file
* Create an inferance function

### 3.4 API
* At the moment the whole project is monolithic. We can deploy this API as a service but it won't be able to identify the pre-trained models
  * Export and load the models from an S3 bucket
* Creat a new endpoind to re-train/update the current model
  * Create model versions/snapshots
  * Run tests to check the coherance of the new model and based on a threshold decide which model to keep ( after we update the model
  )

