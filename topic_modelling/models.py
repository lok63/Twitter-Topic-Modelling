from topic_modelling.preprocessor_all import load_data
from topic_modelling.pipelines import basic_pipeline
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import EnsembleLda, HdpModel, CoherenceModel, Nmf
import matplotlib.pyplot as plt
from gensim.utils import tokenize
from utils import timing
from abc import ABC, abstractmethod
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import gensim
import pandas as pd
import re

RANDOM_STATE = 42

class TopicModel(ABC):

    @abstractmethod
    def fit(self, df:pd.DataFrame, column: str):
        pass

    @abstractmethod
    def train(self, num_topics: int, passes: int):
        pass

    @timing
    def createid2word_dictionary(self):
        """Creates a dictionary that maps words to integers"""
        self.id2word = gensim.corpora.Dictionary(self.df.tolist())
        print(f":: Size of id2word: {len(self.id2word)}")

    @timing
    def filter_extremes(self, no_below=2, no_above=0.5, keep_n=100000):
        """
        no_below : int, optional
            Keep tokens which are contained in at least `no_below` documents.
        no_above : float, optional
            Keep tokens which are contained in no more than `no_above` documents
            (fraction of total corpus size, not an absolute number).
        keep_n : int, optional
            Keep only the first `keep_n` most frequent tokens.
        keep_tokens : iterable of str
            Iterable of tokens that **must** stay in dictionary after filtering.
        """
        self.id2word.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)

    @timing
    def create_bow_coprpus(self):
        self.corpus = [self.id2word.doc2bow(d) for d in self.df.tolist()]


    def get_topics(self):
        words = [re.findall(r'"([^"]*)"', t[1]) for t in self.model.print_topics()]
        topics = [' '.join(t[0:10]) for t in words]
        # Getting the topics
        for id, t in enumerate(topics):
            print(f"------ Topic {id} ------")
            print(t, end="\n\n")

    @timing
    def get_perplexity(self):
        perplexity = self.model.log_perplexity(self.corpus)
        print('\nPerplexity: ', perplexity)
        return perplexity
    @timing
    def get_coherance(self):
        coherenceModel = CoherenceModel(model=self.model, texts=self.df, dictionary=self.id2word, coherence='c_v')
        coherance_score = coherenceModel.get_coherence()
        print('\nCoherence Score: ', coherance_score)
        return coherance_score

    def visualise_topics_notebook(self):
        vis = gensimvis.prepare(topic_model=self.model, corpus=self.corpus, dictionary=self.id2word)
        pyLDAvis.enable_notebook()
        pyLDAvis.display(vis)

    def visualise_topics(self):
        vis_data = gensimvis.prepare(self.model, self.corpus, self.id2word)
        return vis_data

    def compute_coherence_values(self, start=2, limit=10, step=2, plot=False):
        """
        Code retrieved fro: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#17howtofindtheoptimalnumberoftopicsforlda
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = LdaMulticore(corpus=self.corpus, num_topics=num_topics, id2word=self.id2word, random_state=RANDOM_STATE)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=self.df, dictionary=self.id2word, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        if plot:
            x = range(start, limit, step)
            plt.plot(x, coherence_values)
            plt.xlabel("Num Topics")
            plt.ylabel("Coherence score")
            plt.legend(("coherence_values"), loc='best')
            plt.show()
        return model_list, coherence_values


    def compute_coherence_for_topics_a_d(self, k, a, d, c, p):
        lda_model = gensim.models.LdaMulticore(corpus=self.corpus,
                                               id2word=self.id2word,
                                               num_topics=k,
                                               random_state=RANDOM_STATE,
                                               chunksize=c,
                                               passes=p,
                                               alpha=a,
                                               eta="auto",
                                               decay=d)

        coherence_model_lda = CoherenceModel(model=lda_model, texts=self.df, dictionary=self.id2word, coherence='c_v')
        return coherence_model_lda.get_coherence()


class BasicModel(TopicModel):

    def __init__(self, *args, **kwargs):
        # invoke TopicModel.__init__
        super().__init__(*args, **kwargs)

    def fit(self, df:pd.DataFrame, column: str):
        self.df = df[column]


    @timing
    def train(self, num_topics: int = 10, passes: int = 1, chunksize=100, eval_every=10, alpha='asymmetric', eta="auto", decay=0.5):
        self.createid2word_dictionary()
        self.filter_extremes()
        self.create_bow_coprpus()

        print(f"----> Training {self.__class__.__name__} <----")
        base_model = LdaMulticore(corpus=self.corpus,
                                  id2word=self.id2word,
                                  workers=None,
                                  num_topics=num_topics,
                                  passes=passes,
                                  chunksize = chunksize,
                                  eval_every = eval_every,
                                  random_state=RANDOM_STATE,
                                  alpha=alpha,
                                  eta=eta,
                                  decay=0.5)
        self.model = base_model

class NMFModel(TopicModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, df: pd.DataFrame, column: str):
        self.df = df[column]

    @timing
    def train(self,   num_topics=10, passes: int = 10, chunksize=100, eval_every=10):
        self.createid2word_dictionary()
        self.filter_extremes()
        self.create_bow_coprpus()

        print(f"----> Training {self.__class__.__name__} <----")
        base_model = Nmf(corpus=self.corpus,
                                 id2word=self.id2word,
                                 num_topics=num_topics,
                                 passes=passes,
                                 chunksize=chunksize,
                                 eval_every=eval_every,
                                 random_state=RANDOM_STATE
                         )

        self.model = base_model



class MalletModel(TopicModel):

    pass

class EnsembleModel(TopicModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, df: pd.DataFrame, column: str):
        self.df = df[column]

    @timing
    def train(self,   num_topics=10, num_models=4, passes: int = 10, chunksize=100, eval_every=10, alpha='asymmetric', eta=None, decay=0.5):
        self.createid2word_dictionary()
        self.filter_extremes()
        self.create_bow_coprpus()

        print(f"----> Training {self.__class__.__name__} <----")
        base_model = EnsembleLda(corpus=self.corpus,
                                 id2word=self.id2word,
                                 workers=None,
                                 num_topics=num_topics,
                                 num_models=num_models,
                                 passes=passes,
                                 chunksize=chunksize,
                                 eval_every=eval_every,
                                 random_state=RANDOM_STATE,
                                 alpha=alpha,
                                 eta=eta,
                                 decay=0.5)

        self.model = base_model

class HierarchicalModel(TopicModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, df:pd.DataFrame, column: str):
        self.df = df[column]

    @timing
    def train(self):
        self.createid2word_dictionary()
        self.create_bow_coprpus()

        print(f"----> Training {self.__class__.__name__} <----")
        base_model = HdpModel(corpus=self.corpus,
                              id2word=self.id2word,
                              random_state=RANDOM_STATE
                              )
        self.model = base_model





if __name__ == '__main__':

    df = load_data()
    df = df.reset_index(drop=True)

    df = basic_pipeline.apply(df, column='cleanBody')

    nmf_model = NMFModel()
    nmf_model.fit(df, 'cleanBody')
    nmf_model.train(num_topics=10)

    print(nmf_model.get_coherance())
    print(nmf_model.get_topics())


    # ensemble_model = EnsembleModel()
    # ensemble_model.fit(df, 'cleanBody')
    # ensemble_model.train(num_topics=10, num_models=10)
    #
    # print(ensemble_model.get_coherance())
    # print(ensemble_model.get_topics())