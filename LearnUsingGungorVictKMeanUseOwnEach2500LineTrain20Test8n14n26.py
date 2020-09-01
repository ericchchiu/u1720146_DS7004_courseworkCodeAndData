# coding: utf-8

#!!!The data file Gungor_2018_VictorianAuthorAttribution_data-train.csv can be obtained from https://archive.ics.uci.edu/ml/machine-learning-databases/00454/

##input data
import pandas as pd

#note: use your own path
path_to_datafile = '..//..//DS7004//u1720146_DS7004_courseworkCodeAndData//preparationWorks//fromDS7003_Gungor2018VictorianAuthorAttribution_NGram//Gungor_2018_VictorianAuthorAttribution_data-train.csv'
pathToGungorVict = path_to_datafile
gungorVictRow = pd.read_csv(pathToGungorVict, encoding = 'ISO-8859-1')

##form training data (2500 lines x 3) and test data (20 x 3)
##each line about 1000 words
#Use three authors' data:
#author:8 Charles Dickens total lines: 6914/ 14 George Eliot 2696/ 26 Jane Austen 4441
#each first 2500 lines for training, last 20 lines for testing. Each line has 1000 words
for i in [14, 26, 8]:
    allLines = gungorVictRow.loc[gungorVictRow['author'] == i]
    lines2500 = allLines.iloc[0:2500]
    linesLast20 = allLines.iloc[-20:]
    try:
        train = train.append(lines2500)
        test = test.append(linesLast20)
    except:
        train = lines2500
        test = linesLast20
train = train.sample(frac=1, random_state=42).reset_index(drop = True) #7500 lines suffled
test = test.sample(frac=1, random_state=42).reset_index(drop = True) #60 lines suffled

## Import various modules for forming a string cleaning function
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def text_to_wordlist( text, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    text = BeautifulSoup(text).get_text()
    #  
    # 2. Remove non-letters
    text = re.sub("[^a-zA-Z]"," ", text)
    #
    # 3. Convert words to lower case and split them
    words = text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:  #These three lines will not be used. Pleasesee the second parameter of this function
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

## Download the punkt tokenizer and form a sentence splitting function
import nltk.data
#nltk.download() #no need to use this line again after it has been used once  
# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a text into parsed sentences
def text_to_sentences( text, tokenizer, remove_stopwords=False ):
    # Function to split a text into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(text.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call text_to_wordlist to get a list of words
            sentences.append( text_to_wordlist( raw_sentence,               remove_stopwords )) #defined as false in text_to_wordlist
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

#function for parsing the training set
def parsing_sentence_set(text_df):
    sentences = []  # Initialize an empty list of sentences

    print("Parsing sentences from training set")
    for text in text_df["text"]:
        sentences += text_to_sentences(text, tokenizer)
    return sentences

##use the functions to form a cleaned unlabelled training set
##for performming unsupervised learning
sentences = parsing_sentence_set(train)

## Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

# Set values for the single neural network layer's various parameters
#num_features = 300    # Word vector dimensionality                      
#min_word_count = 40   # Minimum word count                        
#num_workers = 4       # Number of threads to run in parallel
#context = 10          # Context window size
#downsampling = 1e-3   # Downsample setting for frequent words

num_features = 300    # Word vector dimensionality                      
min_word_count = 5    # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 6           # Context window size         
downsampling = 1e-3   # Downsample setting for frequent words
epochs= 20             #number of epochs

## Initialize and train the model (this will take some time)
# need to install gensim's word2vec
from gensim.models import word2vec
def form_model_from_sentences(sentences):
    print("Training model...")
    model = word2vec.Word2Vec(sentences, workers=num_workers,                 size=num_features, min_count = min_word_count,                window = context, sample = downsampling, iter = epochs)

    # If you don't plan to train the model any further, calling 
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)
    return model

##form the word2vec model with the training set which will be
##used in the following two methods:
##vector averaging and vector clustering of stop words
model = form_model_from_sentences(sentences)

##check the model
# king - man + woman ~= queen
print(model.most_similar(positive=['king', 'woman'], negative=['man']))

##check the model
# better - good + bad ~= worse
model.most_similar(positive=['better', 'bad'], negative=['good'])

# ****************************************************************
##first method: vector averaging of stop words:
import gensim
all_stopwords = set(gensim.parsing.preprocessing.STOPWORDS)

#be careful: nword and counter must be integers --Chiu
import numpy as np  # Make sure that numpy is imported

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph which are stop words
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    index2word_set2 = all_stopwords
    #
    # Loop over each word in the text and, if it is in the model's
    # vocaublary and is a stop word add its feature vector to the total
    for word in words:
        if word in index2word_set: #and word in index2word_set2: 
            if word in index2word_set2:
                nwords = nwords + 1
                featureVec = np.add(featureVec, model[word])
    # 
    # Divide the result by the number of words to get the average
    if nwords == 0:
        nwords = 1 #avoid devided by zero (i.e. no stop word)
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(texts, model, num_features):
    # Given a set of texts (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    textFeatureVecs = np.zeros((len(texts),num_features),dtype="float32")
    # 
    # Loop through the texts
    for text in texts:
       #
       # Print a status message every 100th text
        if counter%100 == 0:
            haha = counter; hihi = len(texts)
            print(f"Text {haha} of {hihi}") #% (counter, len(texts))
       # 
       # Call the function (defined above) that makes average feature vectors
        #textFeatureVecs[counter] = makeFeatureVec(text, model, num_features)
        textFeatureVecs[counter] = makeFeatureVec(text, model, num_features)
       # Increment the counter
        counter = counter + 1
    return textFeatureVecs

# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. 

clean_train_texts = []
for text in train["text"]:
    #clean_train_reviews.append( review_to_wordlist( review, \
        #remove_stopwords=True )) #do not remove stop words
    clean_train_texts.append( text_to_wordlist( text ))

trainDataVecs = getAvgFeatureVecs( clean_train_texts, model, num_features )

print("Creating average feature vecs for test texts")
clean_test_texts = []
for text in test["text"]:
    #clean_test_texts.append( text_to_wordlist( review, remove_stopwords=True ))
    clean_test_texts.append( text_to_wordlist( text ))

testDataVecs = getAvgFeatureVecs( clean_test_texts, model, num_features )

# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

print("Fitting a random forest to labeled training data...")
forest = forest.fit( trainDataVecs, train["author"] )

# Test & extract results 
result = forest.predict( testDataVecs )

# Write the test results 
output = pd.DataFrame( data={"true_author":test["author"], "pred_author":result} )
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )

confusion_matrix = pd.crosstab(output['true_author'], output['pred_author'], rownames=['Actural'], colnames=['Predicted'])
print('Confusion matrix:\n', confusion_matrix)

from sklearn import metrics
print('Accuracy: ', metrics.accuracy_score(output['true_author'], output['pred_author']))

print('f1 score: ', metrics.f1_score(output['true_author'], output['pred_author'], average = 'weighted'))

# ****************************************************************
##second method: vector clustering of stop words (use KMeans):

from sklearn.cluster import KMeans
import time

start = time.time() # Start time (several to tens of minutes)

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = word_vectors.shape[0] / 5

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = int(num_clusters) )
idx = kmeans_clustering.fit_predict( word_vectors )

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print("Time taken for K Means clustering: ", elapsed, "seconds.")

# Create a Word / Index dictionary, mapping each vocabulary word to
#a cluster number

word_centroid_map = dict(zip( model.wv.index2word, idx ))

# For the first 10 clusters
for cluster in range(0,10):
    #
    # Print the cluster number  
    #print "\nCluster %d" #% cluster
    print(f"\nCluster {cluster}")
    #
    # Find all of the words for that cluster number, and print them out
    a_view = word_centroid_map.items()
    tuples = list(a_view)
    words = []
    for i in range(0,len(word_centroid_map.values())):
        if( tuples[i][1] == cluster ):
            words.append(tuples[i][0])
    print(words)

def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map and word in all_stopwords:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids

# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( (train["text"].size, int(num_clusters)),     dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for text in clean_train_texts:
    train_centroids[counter] = create_bag_of_centroids( text,         word_centroid_map )
    counter += 1

# Repeat for test reviews 
test_centroids = np.zeros((test["text"].size, int(num_clusters)),     dtype="float32" )

counter = 0
for text in clean_test_texts:
    test_centroids[counter] = create_bag_of_centroids( text,         word_centroid_map )
    counter += 1

# This cell take some minutes
# Fit a random forest and extract predictions 
forest = RandomForestClassifier(n_estimators = 100)

# Fitting the forest may take a few minutes
print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids,train["author"])
result = forest.predict(test_centroids)

# Write the test results 
output = pd.DataFrame(data={"true_author":test["author"], "pred_author":result})
output.to_csv( "BagOfCentroidsAuthor.csv", index=False, quoting=3 )

confusion_matrix = pd.crosstab(output['true_author'], output['pred_author'], rownames=['Actural'], colnames=['Predicted'])
print('Confusion matrix:\n', confusion_matrix)

from sklearn import metrics
print('Accuracy: ', metrics.accuracy_score(output['true_author'], output['pred_author']))

print('f1 score: ', metrics.f1_score(output['true_author'], output['pred_author'], average = 'weighted'))

