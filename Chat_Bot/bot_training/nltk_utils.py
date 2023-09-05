import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)
    

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

#Tokenization: Used to split sentences into seperate words for analysis and processing 
#sentence = "See you later, thanks for visiting"
#sentence = tokenize(sentence)
#print(sentence)

#Word Stemming: Essentially cuts the word to better pass through NLP model
#words =  ["organize", "organizes", "organizing"]
#stem_words = [stem(w) for w in words]
#print(stem_words)

#Bag of Words: Comes after tokenization and stemming, reduces words to 1s and 0s to represent text by 
#counting the frequency of individual words, ignoring their order.
#It's used for text analysis and helps convert text into a numerical format for machine learning.
#sentence = ["hello", "how", "are", "you"]
#words = ["hi", "hello", "i", "you", "bye", "thank", "cool"]
#bog = bag_of_words(sentence, words)
#print(bog)