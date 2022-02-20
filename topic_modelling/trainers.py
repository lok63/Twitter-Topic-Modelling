from topic_modelling.pipelines import basic_pipeline
from topic_modelling.preprocessor_all import load_data
from topic_modelling.models import BasicModel

if __name__ == '__main__':
    df = load_data()
    df = basic_pipeline.apply(df, column='cleanBody')


    basic_model = BasicModel()
    basic_model.fit(df,'cleanBody')
    basic_model.train(num_topics=10, passes=10)


