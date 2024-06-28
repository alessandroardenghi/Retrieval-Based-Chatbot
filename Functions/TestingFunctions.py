from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from Functions.PreprocessingFunctions import BOW_discretizer, TFIDF_discretizer

def cosine_similarity_score(train_matrix, test_matrix, train_df, test_df):
    
    result_matrix = cosine_similarity(test_matrix, train_matrix)
    
    result_vector = np.argmax(result_matrix, axis = 1)
    
    data = pd.DataFrame({'user_prompt': test_df['user_prompt'],
                     'model_response': test_df['model_response'],
                     'training_prompt': [train_df['user_prompt'].iloc[i] for i in result_vector],
                     'retrieved_response': [train_df['model_response'].iloc[i] for i in result_vector]})

    data = data.dropna(subset=['model_response', 'retrieved_response'])
    
    smoothingfunction = SmoothingFunction() 

    data['bleu_score'] = data.apply(lambda x: sentence_bleu([x['model_response'].split()], x['retrieved_response'].split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothingfunction.method3), axis=1)
    average = data['bleu_score'].mean()
    
    return data, average


def BOW_exhaustive_search(train, dev, n_gram_params, analyzer, binary, train_df, dev_df):
    results = []
    """ This function takes as input tokenized inputs """
    for i in n_gram_params:
        for j in analyzer:
            for l in binary:
                try:
                    train_matrix, dev_matrix = BOW_discretizer(train, dev, j, i, 0.00, 1.00, l)
                    _, average = cosine_similarity_score(train_matrix, dev_matrix, train_df, dev_df)
                except ValueError:
                    average = 0.00
                results.append({'n_gram' : i, 'analyzer': j, 'binary': l, 'score': average})
    return results[np.argmax(np.array([i['score'] for i in results]))]

def TFIDF_exhaustive_search(train, dev, n_gram_params, train_df, dev_df):
    results = []

    for i in n_gram_params:
        try:
            train_matrix, dev_matrix = TFIDF_discretizer(train, dev, 'char', i, 0.00, 1.00, False, True, True)
            _, stemmed_average = cosine_similarity_score(train_matrix, dev_matrix, train_df, dev_df)
        except ValueError:
            stemmed_average = 0.00
        results.append({'n_gram' : i, 'binary': True, 'smooth': True, 'analyzer':'char', 'score': stemmed_average})
    return results[np.argmax(np.array([i['score'] for i in results]))]

def get_track1_predictions(train, test, train_df, test_df):

    import pandas as pd
    train_data, test_data = TFIDF_discretizer(train, test, 'char', (2, 4), 0.00, 1.00, False, True, True)
    result_matrix = cosine_similarity(test_data, train_data)   
    result_vector = np.argmax(result_matrix, axis = 1)
    data = pd.DataFrame({'test_prompt': test_df['user_prompt'],
                     'training_prompt': [train_df['user_prompt'].iloc[i] for i in result_vector],
                     'retrieved_response': [train_df['model_response'].iloc[i] for i in result_vector], 
                     'conversation_id': test_df['conversation_id'], 
                     'response_id': [train_df['conversation_id'].iloc[i] for i in result_vector]})
    return data

def get_track2_predictions(model, train, test, train_df, test_df):

    from Functions.PreprocessingFunctions import apply_word2vec_model_single
    import pandas as pd
    train_data = apply_word2vec_model_single(model, train)
    test_data = apply_word2vec_model_single(model, test)
    result_matrix = cosine_similarity(test_data, train_data)   
    result_vector = np.argmax(result_matrix, axis = 1)
    data = pd.DataFrame({'test_prompt': test_df['user_prompt'],
                     'training_prompt': [train_df['user_prompt'].iloc[i] for i in result_vector],
                     'retrieved_response': [train_df['model_response'].iloc[i] for i in result_vector], 
                     'conversation_id': test_df['conversation_id'], 
                     'response_id': [train_df['conversation_id'].iloc[i] for i in result_vector]})
    return data

def get_track3_predictions(train, test, train_df, test_df):
    result_matrix = cosine_similarity(test, train)
    result_vector = np.argmax(result_matrix, axis = 1)
    data = pd.DataFrame({'test_prompt': test_df['user_prompt'],
                     'training_prompt': [train_df['user_prompt'].iloc[i] for i in result_vector],
                     'retrieved_response': [train_df['model_response'].iloc[i] for i in result_vector], 
                     'conversation_id': test_df['conversation_id'], 
                     'response_id': [train_df['conversation_id'].iloc[i] for i in result_vector]})
    return data
    