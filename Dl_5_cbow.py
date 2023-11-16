# -*- coding: utf-8 -*-
"""DL_5_CBOW.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fr43yY9G_EW0TyXVONrRktKOSCOHbfk0
"""

#Import Required Libraries
import nltk # Powerful library for working with human language data
from nltk.corpus import brown # Brown corpus is a collection of text samples that is often used for training and testing in NLP.
from gensim.models import Word2Vec #Gensim is used for topic modelling and document similarity analysis.

#Data Preprocessing
#We download the brown corpus in this section and use it as the sample data.
nltk.download('brown')  #Download Brown COrpus
data = brown.sents()  # Use the Brown corpus from NLTK as sample data #load Brown Corpus

#CBOW model using the gensim library's Word2Vec class
model = Word2Vec(data, min_count=1,  window=5)   # CBOW model using the gensim library's Word2Vec
#data :This is the input data for training the Word2Vec model. In your case, data should be a collection of sentences or a list of words.(from Brown)
#min_count=1 sets the minimum count of occurrences for a word to be considered during training. Words that occur less than this specified count
# are usually ignored. (set to 1 to include all words)
# This parameter sets the maximum distance between the current and predicted word within a sentence.In other words, it defines the size of context window.
#A window of 5 means that the model will consider five words to the left and five words to the right of the target word as context.

#train
model.train(data, total_examples=len(data), epochs=5) # Model is trained on the data with a specified number of epochs
#Continue training of Word2Vec Model
# data : Input data containing sentences from Brown Corpus
# total_examples: Total number of sentences in dataset for better training
# epochs : number of iterations over dataset during training

print(data)

word_vectors = model.wv  #model in this section learns the embeddings.
#extracts the Word Vectors from the trained Word2Vec model (model). In gensim's Word2Vec implementation, the wv attribute holds the Word Vectors.

#we calculate the cosine similarity between two words, 'woman' and 'man',
# This measures how similar the two words are in meaning.

similarity = word_vectors.similarity('man', 'woman')
print(f"Similarity between 'woman' and 'man': {similarity}")