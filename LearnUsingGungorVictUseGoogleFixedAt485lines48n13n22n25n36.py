# coding: utf-8

#need to download GoogleNews-vectors-negative300.bin first
from gensim.models import Word2Vec, KeyedVectors
#use your path!!!
pathToGoogleNews300 = '..//fromBlogOfShaneLynnWordEmbeddingsWithSpacyAndGensim_GoogleNews300//data//GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(pathToGoogleNews300, binary=True)

import pandas as pd
#need to download Gungor_2018_VictorianAuthorAttribution_data-train.csv from http://archive.ics.uci.edu/ml/datasets/Victorian+Era+Authorship+Attribution
#use your path!!!
pathToGungorVict = '..//..//DS7004//u1720146_DS7004_courseworkCodeAndData//preparationWorks//fromDS7003_Gungor2018VictorianAuthorAttribution_NGram//Gungor_2018_VictorianAuthorAttribution_data-train.csv'
gungorVictRow = pd.read_csv(pathToGungorVict, encoding = 'ISO-8859-1')
#48: Washington Irving/ 13: Frances Hodgson Burnett/ 22: Jacob Abbott/ 25: James Payn/ 36: Oliver Optic

for i in [13, 22, 25, 36, 48]:
    allLines = gungorVictRow.loc[gungorVictRow['author'] == i]
    lines350 = allLines.iloc[0:350]
    linesLast20 = allLines.iloc[-20:]
    try:
        train = train.append(lines350)
        test = test.append(linesLast20)
    except:
        train = lines350
        test = linesLast20
train = train.sample(frac=1, random_state=42).reset_index(drop = True) #1750 lines suffled
test = test.sample(frac=1, random_state=42).reset_index(drop = True) #100 lines suffledtest = authorsLines.sample(frac=0.3, random_state=42)

print(model.most_similar(positive=['king', 'woman'], negative=['man']))

print(model.most_similar(positive=['better', 'bad'], negative=['good']))

print(model.most_similar(positive=['lead', 'saw'], negative=['led']))

print(model.doesnt_match("man woman child kitchen".split()))

import gensim
all_stopwords = set(gensim.parsing.preprocessing.STOPWORDS)

#be careful: nword and counter must be integers --Chiu
import numpy as np  # Make sure that numpy is imported

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
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
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            if word in index2word_set2: 
                nwords = nwords + 1
                featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    if nwords == 0:
        nwords = 1
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
        if counter%50 == 0:
            haha = counter; hihi = len(reviews)
            print(f"Review {haha} of {hihi}") #% (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
        #reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       #
       # Increment the counter
        counter = counter + 1
    return reviewFeatureVecs

# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.
num_features = 300
clean_train_reviews = []
for review in train["text"]:
    #clean_train_reviews.append( review_to_wordlist( review, \
        #remove_stopwords=True ))
    clean_train_reviews.append( review_to_wordlist( review ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

print("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["text"]:
    #clean_test_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
    clean_test_reviews.append( review_to_wordlist( review ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

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
print(confusion_matrix)

from sklearn import metrics
print('Accuracy: ', metrics.accuracy_score(output['true_author'], output['pred_author']))

print('f1 score: ', metrics.f1_score(output['true_author'], output['pred_author'], average = 'weighted'))

