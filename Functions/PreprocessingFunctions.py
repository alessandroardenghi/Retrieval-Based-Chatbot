
import spacy
from nltk import SnowballStemmer
import numpy as np

def tokenize(corpus):
    """ This function takes a list of sentences as input"""
    import re
    temp =  [np.array([word.lower() for word in re.findall(r'\w+|\S|=', sentence)]) for sentence in corpus]
    tokenized_corpus =  [[s for s in sentence if s != ''] for sentence in temp]
    return tokenized_corpus

def lemmatization(corpus):
    
    import spacy
    nlp = spacy.load("en_core_web_sm")

    documents = list(nlp.pipe(corpus))
    lemmatized_corpus = [[token.lower().lemma_ for token in document] for document in documents]
    
    return lemmatized_corpus

def lemmatization_and_stemming(dataset):
    
    nlp = spacy.load("en_core_web_sm")
    stemmer = SnowballStemmer('english')
    
    documents = list(nlp.pipe(dataset))

    lemmatized_corpus = [[token.lemma_ for token in document] for document in documents]
    tokens = [[token.text for token in sentence] for sentence in documents]
    stemmed_corpus = [[stemmer.stem(token) for token in document] for document in tokens]
    
    return lemmatized_corpus, stemmed_corpus

def remove_punctuation(tokenized_corpus):
    
    """ This function requires the corpus to already be tokenized"""
    
    import string
    no_punct_corpus = [[token.lower() for token in sentence if token not in string.punctuation] for sentence in tokenized_corpus]
    
    return no_punct_corpus

def remove_stopwords(tokenized_corpus, stop_words):
    
    """ This function requires the corpus to already be tokenized """
    
    no_sw_corpus = [[token.lower() for token in sentence if token not in stop_words] for sentence in tokenized_corpus]

    return no_sw_corpus



def apply_BOW(list_of_arguments):
    result = {}
    for key in list_of_arguments:
        list1, list2 = BOW_discretizer(list_of_arguments[key][0], list_of_arguments[key][1], 'word', (1, 1), 0.00, 1.00, False)
        result[key] = [list1, list2]
    return result

def apply_cosine_similarity(list_of_arguments, train_df, dev_df):
    from Functions.TestingFunctions import cosine_similarity_score
    result = {}
    for key in list_of_arguments:
        data, average = cosine_similarity_score(list_of_arguments[key][0], list_of_arguments[key][1], train_df, dev_df)
        result[key] = [data, average]
    return result

def apply_TFIDF(list_of_arguments):
    result = {}
    for key in list_of_arguments:
        list1, list2 = TFIDF_discretizer(list_of_arguments[key][0], list_of_arguments[key][1], 'word', (1, 1), 0.00, 1.00, False, False, False)
        result[key] = [list1, list2]
    return result
   

def BOW_discretizer(training_corpus, test_corpus, analyzer, n_gram_range, minimum, maximum, binary):
    
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(analyzer = analyzer, 
                                ngram_range= n_gram_range, 
                                min_df=minimum,
                                max_df=maximum, 
                                binary = binary,
                                stop_words='english')
    training_corpus = [" ".join(sentence) for sentence in training_corpus]
    test_corpus = [" ".join(sentence) for sentence in test_corpus]
    train_matrix = vectorizer.fit_transform(training_corpus)
    test_matrix = vectorizer.transform(test_corpus)
    
    return train_matrix, test_matrix

def TFIDF_discretizer(training_corpus, test_corpus, analyzer, n_gram_range, minimum, maximum, lowercase, binary, smooth_idf):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(analyzer = analyzer, 
                                ngram_range= n_gram_range, 
                                min_df=minimum,
                                max_df=maximum, 
                                lowercase = lowercase,
                                binary = binary,
                                smooth_idf = smooth_idf,
                                stop_words='english')

    training_corpus = [" ".join(sentence) for sentence in training_corpus]
    test_corpus = [" ".join(sentence) for sentence in test_corpus]
    
    train_matrix = vectorizer.fit_transform(training_corpus)
    test_matrix = vectorizer.transform(test_corpus)

    return train_matrix, test_matrix


def apply_fasttext_model_single(model, corpus):
    
    """ This function requires the corpus to be tokenized"""
    
    list_of_vectorized_words = [np.array([model.get_word_vector(word.lower()) for word in sentence]) for sentence in corpus]
    
    for i, row in enumerate(list_of_vectorized_words):
        if len(row) == 0:
            list_of_vectorized_words[i] = [np.zeros((300,))]
            
    vectorized_docs = [np.mean(vectors, axis = 0) for vectors in list_of_vectorized_words]
    vectorized_docs = np.vstack(vectorized_docs)
    
    return vectorized_docs

def apply_fasttext_model_multiple(model, list_of_arguments):
    result = {}
    for key in list_of_arguments:
        list1 = apply_fasttext_model_single(model, list_of_arguments[key][0])
        list2 = apply_fasttext_model_single(model, list_of_arguments[key][1])
        result[key] = [list1, list2]
    return result

def apply_word2vec_model_multiple(model, list_of_arguments):
    result = {}
    for key in list_of_arguments:
        list1 = apply_word2vec_model_single(model, list_of_arguments[key][0])
        list2 = apply_word2vec_model_single(model, list_of_arguments[key][1])
        result[key] = [list1, list2]
    return result


def get_word_vector(model, word):
    try:
        return model[word.lower()]
    except KeyError:

        return None
    
def apply_word2vec_model_single(model, corpus):
    
    """ This function requires the corpus to be tokenized"""
    
    list_of_vectorized_words = [np.array([get_word_vector(model, word.lower()) for word in sentence if get_word_vector(model, word) is not None]) for sentence in corpus]
    
    # Substituting empty sentences with zero vector
    for i, row in enumerate(list_of_vectorized_words):
        if len(row) == 0:
            list_of_vectorized_words[i] = [np.zeros((300,))]
            
    vectorized_docs = [np.mean(vectors, axis = 0) for vectors in list_of_vectorized_words]
    vectorized_docs = np.vstack(vectorized_docs)
    
    return vectorized_docs

