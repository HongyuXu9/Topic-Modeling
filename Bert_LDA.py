from collections import Counter
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import os
     

def get_topic_words(token_lists, labels, k=None):
  ''' Get topic within each topic form clustering results '''
  if k is None:
    k = len(np.unique(labels))
  topics = ['' for _ in range(k)]
  for i, c in enumerate(token_lists):
    topics[labels[i]] += (' ' + ' '.join(c))
  word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
  # get sorted word counts
  word_counts = list(map(lambda x: sorted(x, key=lambda x: x[1], reverse=True),word_counts))
  # get topics
  topics = list(map(lambda x: list(map(lambda x: x[0], x[:10])), word_counts))

  return topics

def get_coherence_cv(model, token_lists, measure='c_v'):
  ''' Get model coherence from gensim.models.coherencemodel
  : param model: Topic_Model object
  : param token_lists: token list of docs
  : param topics: topics as top words 
  : param measure: coherence metrics
  : return: coherence score '''

  if model.method == 'LDA':
    cm = CoherenceModel(model=model.ldamodel, texts=token_lists, corpus = model.corpus, dictionary=model.dictionary, coherence = measure)
  else:
    topics = get_topic_words(token_lists, model.cluster_model.labels_)
    cm = CoherenceModel(topics=topics, texts = token_lists, corpus=model.corpus, dictionary=model.dictionary, coherence = measure)
    return cm.get_coherence()

def get_coherence_npmi(model, token_lists, measure='c_npmi'):
  ''' Get model coherence from gensim.models.coherencemodel
  : param model: Topic_Model object
  : param token_lists: token list of docs
  : param topics: topics as top words 
  : param measure: coherence metrics
  : return: coherence score '''

  if model.method == 'LDA':
    cm = CoherenceModel(model=model.ldamodel, texts=token_lists, corpus = model.corpus, dictionary=model.dictionary, coherence = measure)
  else:
    topics = get_topic_words(token_lists, model.cluster_model.labels_)
    cm = CoherenceModel(topics=topics, texts = token_lists, corpus=model.corpus, dictionary=model.dictionary, coherence = measure)
    return cm.get_coherence()
      
def get_silhoulette(model):
  ''' Get silhoulette score from model
  :param_model: Topic_model score
  :return: silhoulette score '''
  if model.method == 'LDA':
    return
  lbs = model.cluster_model.labels_
  vec = model.vec[model.method]
  return silhoulette_score(vec, lbs)

def plot_project(embedding, lbs):
  '''
  Plot UMAP embeddings
  :param embedding: UMAP (or other) embeddings
  :param lbs: labels
  '''
  n = len(embedding)
  counter = Counter(lbs)
  for i in range(len(np.unique(lbs))):
    plt.plot(embedding[:, 0][lbs == i], embedding[:, 1][lbs == i], '.', alpha=0.5, label='cluster {}: {:.2f}%'.format(i, counter[i] / n*100))
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='-', linewidth=0.25)

def visualize(model):
  '''
  Visualize the result for the topic model by 2D embedding (UMAP)
  :param model: Topic_Model object
  '''
  if model.method == 'LDA':
    return
  reducer = umap.UMAP()
  print('Calculating UMAP projection...')
  vec_umap = reducer.fit_transform(model.vec[model.method])
  print('Calculating the Umap projection. Done!')
  plot_project(vec_umap, model.cluster_model.labels_)

def get_wordcloud(model, token_lists, topic):
    print(f'Getting wordcloud for topic {topic} ...')
    lbs = model.cluster_model.labels_
    
    # Filtra le liste in base all'argomento
    filtered_tokens = [token_list for i, token_list in enumerate(token_lists) if lbs[i] == topic and isinstance(token_list, list)]
    
    # Assicurati che ogni elemento nella lista sia una stringa
    tokens = ' '.join([' '.join(map(str, token_list)) for token_list in filtered_tokens if token_list])  # Evita liste vuote

    # Verifica se tokens non è vuoto
    if tokens:
        wordcloud = WordCloud(width=800, height=560, 
                              background_color='white', collocations=False, 
                              min_font_size=10).generate(tokens)
        
        # Plot the WordCloud image
        plt.figure(figsize=(8, 8), facecolor=None) 
        plt.imshow(wordcloud) 
        plt.axis("off") 
        plt.tight_layout(pad=0) 
        plt.show()
    else:
        print(f"No tokens found for topic {topic}.")

import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
     

class Autoencoder:
    """
    Autoencoder for learning latent space representation
    architecture simplified for only one hidden layer
    """
    def __init__(self, latent_dim = 32, activation='relu', epochs=200, batch_size=128):
        self.latent_dim = latent_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.his = None

    def _compile(self, input_dim):
        '''
        compile the computational graph
        '''
        input_vec = Input(shape=(input_dim,))
        encoded = Dense(self.latent_dim, activation=self.activation)(input_vec)
        decoded = Dense(input_dim, activation=self.activation)(encoded)
        self.autoencoder = Model(input_vec, decoded)
        self.encoder = Model(input_vec, encoded)
        encoded_input = Input(shape=(self.latent_dim,))
        decoded_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, self.autoencoder.layers[-1](encoded_input))
        self.autoencoder.compile(optimizer='adam', loss=keras.losses.mean_absolute_error)

    def fit(self, X):
        if not self.autoencoder:
            self._compile(X.shape[1])
            X_train, X_test = train_test_split(X)
            self.his = self.autoencoder.fit(X_train, X_train, epochs=200, batch_size=128, shuffle=True, validation_data=(X_test, X_test), verbose=0)
     

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim import corpora
import gensim
import numpy as np
#from Autoencoder import *
#from preprocess import *
from datetime import datetime
     

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim import corpora
import gensim
import numpy as np
#from Autoencoder import *
#from preprocess import *
from datetime import datetime

def get_token_lists(docs):
    token_lists = []  # word level token lists
    for doc in docs:
        token_list = doc.split()  # Dividi la frase in parole (token)
        if token_list:
            token_lists.append(token_list)
    return token_lists

# define model object
class Topic_Model:
    def __init__(self, k=10, method='TFIDF'):
        """
        :param k: number of topics
        :param method: method chosen for the topic model
        """
        if method not in {'TFIDF', 'LDA', 'BERT', 'LDA_BERT'}:
            raise Exception('Invalid method!')
        self.k = k
        self.num_topics = k
        self.dictionary = None
        self.corpus = None
        #         self.stopwords = None
        self.cluster_model = None
        self.ldamodel = None
        self.vec = {}
        self.gamma = 15  # parameter for reletive importance of lda
        self.method = method
        self.AE = None
        self.id = method + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def vectorize(self, sentences, token_lists, method=None):
        """
        Get vecotr representations from selected methods
        """
        # Default method
        if method is None:
            method = self.method

        # turn tokenized documents into a id <-> term dictionary
        self.dictionary = corpora.Dictionary(token_lists)
        # convert tokenized documents into a document-term matrix
        self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        if method == 'TFIDF':
            print('Getting vector representations for TF-IDF ...')
            tfidf = TfidfVectorizer()
            vec = tfidf.fit_transform(sentences)
            print('Getting vector representations for TF-IDF. Done!')
            return vec

        elif method == 'LDA':
            print('Getting vector representations for LDA ...')
            if not self.ldamodel:
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=20)

            def get_vec_lda(model, corpus, k):
                """
                Get the LDA vector representation (probabilistic topic assignments for all documents)
                :return: vec_lda with dimension: (n_doc * n_topic)
                """
                n_doc = len(corpus)
                vec_lda = np.zeros((n_doc, k))
                for i in range(n_doc):
                    # get the distribution for the i-th document in corpus
                    for topic, prob in model.get_document_topics(corpus[i]):
                        vec_lda[i, topic] = prob

                return vec_lda

            vec = get_vec_lda(self.ldamodel, self.corpus, self.k)
            print('Getting vector representations for LDA. Done!')
            return vec

        elif method == 'BERT':

            print('Getting vector representations for BERT ...')
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('bert-base-nli-max-tokens')
            vec = np.array(model.encode(sentences, show_progress_bar=True))
            print('Getting vector representations for BERT. Done!')
            return vec

             
        elif method == 'LDA_BERT':
        #else:
            vec_lda = self.vectorize(sentences, token_lists, method='LDA')
            vec_bert = self.vectorize(sentences, token_lists, method='BERT')
            vec_ldabert = np.c_[vec_lda * self.gamma, vec_bert]
            self.vec['LDA_BERT_FULL'] = vec_ldabert
            if not self.AE:
                self.AE = Autoencoder()
                print('Fitting Autoencoder ...')
                self.AE.fit(vec_ldabert)
                print('Fitting Autoencoder Done!')
            vec = self.AE.encoder.predict(vec_ldabert)
            return vec

    def fit(self, sentences, token_lists, method=None, m_clustering=None):
        """
        Fit the topic model for selected method given the preprocessed data
        :docs: list of documents, each doc is preprocessed as tokens
        :return:
        """
        # Default method
        if method is None:
            method = self.method
        # Default clustering method
        if m_clustering is None:
            m_clustering = KMeans

        # turn tokenized documents into a id <-> term dictionary
        if not self.dictionary:
            self.dictionary = corpora.Dictionary(token_lists)
            # convert tokenized documents into a document-term matrix
            self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        ####################################################
        #### Getting ldamodel or vector representations ####
        ####################################################

        if method == 'LDA':
            if not self.ldamodel:
                print('Fitting LDA ...')
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=20)
                print('Fitting LDA Done!')
        else:
            print('Clustering embeddings ...')
            self.cluster_model = m_clustering(self.k)
            self.vec[method] = self.vectorize(sentences, token_lists, method)
            self.cluster_model.fit(self.vec[method])
            print('Clustering embeddings. Done!')

    def predict(self, sentences, token_lists, out_of_sample=None):
        """
        Predict topics for new_documents
        """
        # Default as False
        out_of_sample = out_of_sample is not None

        if out_of_sample:
            corpus = [self.dictionary.doc2bow(text) for text in token_lists]
            if self.method != 'LDA':
                vec = self.vectorize(sentences, token_lists)
                print(vec)
        else:
            corpus = self.corpus
            vec = self.vec.get(self.method, None)

        if self.method == "LDA":
            lbs = np.array(list(map(lambda x: sorted(self.ldamodel.get_document_topics(x),
                                                     key=lambda x: x[1], reverse=True)[0][0],
                                    corpus)))
        else:
            lbs = self.cluster_model.predict(vec)
        return lbs