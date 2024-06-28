# 🤖 Retrieval-Based Chatbot

## 🌟 Highlights
In this project, I create 3 Retrieval-Based Chatbots using 3 different approaches to represent words and sentences: <br>
- Discrete Representation (TFIDF, BOW, ...)
- Continuous Representation (Word2Vec, FastText, Doc2Vec)
- Transformer Based Representation (Bert, ...)


## ℹ️ Overview

This project aimed at building a Retrieval-Based Chatbot that, when asked a question, would look into its database, find the question with highest semantic similarity to the question asked by the user and returned its answer. <br>
The quality of the returned answer was evaluated using the BLEU Score, which took as input the retrieved answer and the true answer, and computed how similar they were. <br>
To try to maximize this score, I explored 3 different approaches to word and sentence representation, as well as many different approaches to word preprocessing. Moreover, for any given query, I always returned the question which, under some represenation, was closest to it with respect to the cosine similarity measure. The approaches were the following: <br>
- **Discrete Representation**: In this part, I represented sentences using the Bag of Words (BOW) representation and also using the TFIDF representation. To reduce the dimensionality of the output vectors, I also experimented some different combinations of tokenization, punctuation removal, stopword removal, stemming and lemmatization. The results were good, but the semantic content of sentences was not captures, since these methods yield high cosine similarity if sentences share the same words (but not the same semantic meaning)
- **Continuous Representation**: In this second part, I used a pretrained Word2Vec embedding to embed each single word present in the sentence, and I then took the arithmetic mean of the embeddings of the words to obtain a vector embedding for the sentence, which yielded great results. I also tried an analogous approach using pretrained FastText embeddings, which yielded slightly worse results. I finally tried using Doc2Vec, but I did not have enough data for the training and the results were poor.
- **Transformer-Based Representation**: Finally, I used pretrained Transformer Models which were specifically trained for sentence semantic similarity estimation to obtain a vector embedding version of each query. These models seemed to be the best at capturing actual semantic meaning of sentences, and they yielded better results than the two aforementioned approaches. After having experimented with some of these models, I picked the one that performed best, namely **all-mpnet-base-v2**.



### ✍️ Authors

Alessandro Ardenghi
