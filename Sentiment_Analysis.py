# -*- coding: utf-8 -*-
# Sentiment Analysis

# Imports
from pattern.nl import sentiment as sentimentnl
import re
import codecs
import random
import pickle
import math
import numpy as np
np.set_printoptions(threshold=np.nan)
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

# Variables
LOAD_MODEL = 0
N_MODEL_STEPS = 20
LOAD_AGREEMENT = 1
N_AGREEMENT_SAMPLES = 30
FILENAMES = ["dominos","pizzahut","new york pizza"]

# Get a list of tweets from the file
def getTweets(FILENAME):
	file = codecs.open("source\\text_"+FILENAME+".txt", "r", "utf8")
	lines = file.readlines()
	file.close()
	tweets = lines[1::2]
	tweets = [tweet for tweet in tweets if not tweet.startswith("RT")] # Remove RTs
	return tweets
	
# Tokenize a single tweet
def tokenize(tweet):
	tokens = re.findall(r"\w+",tweet) 
	for token in tokens:
		token = token.lower()
	return tokens

# Get n samples per company
def getSamples(n_samples):
	all_samples = []
	for FILENAME in FILENAMES:
		random.seed(FILENAME)
		tweets = getTweets(FILENAME)
		size = len(tweets)
		indices = random.sample(range(size), n_samples)
		samples = [tweets[index] for index in indices]
		all_samples.extend(samples)
	return all_samples
	
# Give sentiment on the samples
def giveSentiment(samples):
	sentiment = []
	for sample in samples:
		print("\n"+sample.encode("utf8"))
		sentiment.append(input("Give sentiment. Pos = 1 - Neg = 2\n"))
	return sentiment

# Calculate Cohen's Kappa
def getKappa(sentiment_1, sentiment_2):
	size = len(sentiment_1)
	# Count scores
	scores = [[0, 0],[0, 0]]
	for i in range(size):
		scores[sentiment_1[i]-1][sentiment_2[i]-1] += 1
	# Calculate total and agreed
	total = size
	agreed = scores[0][0]+scores[1][1]
	# Transpose scores
	scores_T = map(list,map(None,*scores))
	# Calculate expected
	exp0 = float(sum(scores[0])) * sum(scores_T[0]) / total
	exp1 = float(sum(scores[1])) * sum(scores_T[1]) / total
	expected = exp0 + exp1
	Kappa = (agreed-expected)/(total-expected)
	return Kappa

# Get agreement: 0 = disagree, 1 = positive, 2 = negative
def getAgreement(sentiment_1, sentiment_2):
	agreement = [sentiment_1[i] if sentiment_1[i]==sentiment_2[i] else 0 for i in range(len(sentiment_1))]
	return agreement

# Create a test set as a tuple with tweet list and sentiment list
def getTestSet(samples, agreement):
	test_tweets = [samples[i] for i in range(len(agreement)) if agreement[i]>0]
	true_sentiment = [agree for agree in agreement if agree>0]
	return (test_tweets, true_sentiment)

# Run sentiment analysis from pattern.nl on these tweets	
def sentimentAnalysisPattern(tweets):
	sentiment = []
	for tweet in tweets:
		(pol, sub) = sentimentnl(tweet)
		sent =  3 if pol<0 else (1 if pol>0 else 2)
		sentiment.append(sent)
	return sentiment
	
# Calculate macro F1 scores based on true sentiment and sentiment after analysis
def macro_F1_score(sentiment, true_sentiment):
	scores = [[0, 0],[0, 0]]
	size = len(sentiment)
	# Count scores
	for i in range(size):
		scores[sentiment[i]-1][true_sentiment[i]-1] += 1
	print scores
	# Transpose scores
	scores_T = map(list,map(None,*scores))
	# Precision
	precision_0 = float(scores[0][0]) / sum(scores[0])
	precision_1 = float(scores[1][1]) / sum(scores[1])
	macro_precision = (precision_0+precision_1)/2
	# Recall
	recall_0 = float(scores[0][0]) / sum(scores_T[0])
	recall_1 = float(scores[1][1]) / sum(scores_T[1])
	macro_recall = (recall_0+recall_1)/2
	# Harmonic mean
	F1 = (2*macro_precision*macro_recall)/(macro_precision+macro_recall)
	return (F1, macro_precision, macro_recall)

# Let two users rate tweets and save the agreed tweets and sentiments
def saveAgreement(n_samples, agreement, kappa):
	file = open("Sentiment_Agreement", "wb")
	pickle.dump((n_samples, agreement, kappa),file)
	file.close()

# Load agreed tweets and sentiments	
def loadAgreement():
	file = open("Sentiment_Agreement", "rb")
	(n_samples, agreement, kappa) = pickle.load(file)
	file.close()
	return (n_samples, agreement, kappa)	
	
### MY OWN SENTIMENT ANALYSIS BELOW ###
#  Naive Bayes with E-M on Emoticons  #
# Positive emoticons:  :)  :D   :-)   #
# Negative emoticons:  :(  :'(  :-(   #
# No(0) Positive(1) Negative(2) label #
#######################################

def expectationMaximization(n_steps):
	# get all tweets
	all_tweets = []
	for FILENAME in FILENAMES: 
		tweets = getTweets(FILENAME)
		all_tweets.extend(tweets)	
	
	n_tweets = len(all_tweets)
		
	# create vocabulary
	vocabulary = {}
	index = 0
	for tweet in all_tweets:
		tokens = tokenize(tweet)
		for token in tokens:
			if not token in vocabulary:
				vocabulary[token] = index
				index += 1
	n_tokens = index
	
	# Preliminary labels
	positive_labels = []
	negative_labels = []
	
	# tweet token frequency matrix
	tweet_token_freq = lil_matrix((n_tweets,n_tokens),dtype=int)
	tweet_index = 0.0
	for index, tweet in enumerate(all_tweets):
		# tweet token frequency
		tokens = tokenize(tweet)
		for token in tokens:
			token_index = vocabulary[token]
			tweet_token_freq[tweet_index, token_index] += 1			
		tweet_index += 1.0
		# get preliminary labels
		mood = getEmoticons(tweet)
		if mood > 0: positive_labels.append(index) # positive
		if mood < 0: negative_labels.append(index) # negative
	
	print "positive emoticons: "+str(len(positive_labels))
	print "negative emoticons: "+str(len(negative_labels))
	
	# P(token given positive class) or P(t|c+)
	P_token_pos = [0.5]*n_tokens
	# P(token given negative class) or P(t|c-)
	P_token_neg = [0.5]*n_tokens	
	# P(positive class given tweet) or P(c+|d)
	P_tweet_pos = [0.5]*n_tweets
	# P(negative class given tweet) or P(c-|d)
	P_tweet_neg = [0.5]*n_tweets	
	# override preliminary labels to the P_tweet_label
	for index in positive_labels:
		P_tweet_pos[index] = 1.0
		P_tweet_neg[index] = 0.0
	for index in negative_labels:
		P_tweet_pos[index] = 0.0
		P_tweet_neg[index] = 1.0
	# P(c+) prior
	P_prior_pos = (1.0 + len(positive_labels)) / (2.0 + len(positive_labels) + len(negative_labels))
	# P(c-) prior
	P_prior_neg = (1.0 + len(negative_labels)) / (2.0 + len(positive_labels) + len(negative_labels))
	
	for i in range(0,n_steps):
		print "step"
		
		# M-phase
		
		pos_count = (tweet_token_freq.transpose().dot(P_tweet_pos)).transpose()
		neg_count = (tweet_token_freq.transpose().dot(P_tweet_neg)).transpose()
		
		P_token_pos = (1.0+pos_count)/(pos_count+neg_count+2.0) #changed n_tokens to 2.0  since there are only two classes
		P_token_neg = (1.0+neg_count)/(pos_count+neg_count+2.0) #correct... not correct tho
		
		# E-phase
		
		P_tweet_pos_count = [0]*n_tweets
		P_tweet_neg_count = [0]*n_tweets
		indices = tweet_token_freq.nonzero()
		pairs = zip(indices[0], indices[1])
		pos_count_dict = {}
		neg_count_dict = {}
		for (tweet,token) in pairs:
			if tweet in pos_count_dict:
				pos_count_dict[tweet] += math.log(P_token_pos[token])
				neg_count_dict[tweet] += math.log(P_token_neg[token])
			else:
				pos_count_dict[tweet] = math.log(P_token_pos[token]) + math.log(P_prior_pos)
				neg_count_dict[tweet] = math.log(P_token_neg[token]) + math.log(P_prior_neg)
		for tweet in pos_count_dict:
			P_tweet_pos_count[tweet] = pos_count_dict[tweet]
		for tweet in neg_count_dict:
			P_tweet_neg_count[tweet] = neg_count_dict[tweet]
		P_tweet_pos_count = np.asarray(P_tweet_pos_count)
		P_tweet_neg_count = np.asarray(P_tweet_neg_count)
		
		P_tweet_pos = 1.0/(np.exp(P_tweet_neg_count - P_tweet_pos_count)+1.0)
		P_tweet_neg = 1.0/(np.exp(P_tweet_pos_count - P_tweet_neg_count)+1.0)
		
		# override preliminary labels to the P_tweet_label
		for index in positive_labels:
			P_tweet_pos[index] = 1.0
			P_tweet_neg[index] = 0.0
		for index in negative_labels:
			P_tweet_pos[index] = 0.0
			P_tweet_neg[index] = 1.0
		
		# calculate priors
		P_prior_pos = (1.0 + np.sum(P_tweet_pos)) / (2.0 + np.sum(P_tweet_pos) + np.sum(P_tweet_neg))
		P_prior_neg = (1.0 + np.sum(P_tweet_neg)) / (2.0 + np.sum(P_tweet_pos) + np.sum(P_tweet_neg))
		
		model = (vocabulary, P_token_pos.tolist(), P_token_neg.tolist(), P_prior_pos.tolist(), P_prior_neg.tolist())
		return model
	
def getEmoticons(tweet):
	# return the mood of the tweet based on emoticons
	mood = 0
	positive = [":)", ":D", ":-)"]
	negative = [":(", ":'(", ":-(","slecht"]
	for emoticon in positive:
		if emoticon in tweet:
			mood += 1
	for emoticon in negative:
		if emoticon in tweet:
			mood -= 1
	return mood

def sentimentAnalysisNaiveBayes(tweets, model):
	(vocabulary, P_token_pos, P_token_neg, P_prior_pos, P_prior_neg) = model
	sentiment = []
	for tweet in tweets:
		tokens = tokenize(tweet)
		P_pos = math.log(P_prior_pos)
		P_neg = math.log(P_prior_neg)
		for token in tokens:
			if token in vocabulary:
				index = vocabulary[token]
				P_pos += math.log(P_token_pos[index])
				P_neg += math.log(P_token_neg[index])
		odds = 1 / (math.exp(P_neg - P_pos) + 1)
		sen = 1 if odds>0.5 else 2
		sentiment.append(sen)
	return sentiment
	
def saveModel(model):
	file = open("Sentiment_Model", "wb")
	pickle.dump(model,file)
	file.close()

def loadModel():
	file = open("Sentiment_Model", "rb")
	model = pickle.load(file)
	file.close()
	return model	

### MAIN ###

if LOAD_AGREEMENT==1:
	# Load agreement
	(N_AGREEMENT_SAMPLES, agreement, kappa) = loadAgreement()
	# Get sample tweets
	samples = getSamples(N_AGREEMENT_SAMPLES)
	print kappa
	print agreement
else: 
	# Create agreement
	samples = getSamples(N_AGREEMENT_SAMPLES)
	sentiment_1 = giveSentiment(samples)
	sentiment_2 = giveSentiment(samples)
	agreement = getAgreement(sentiment_1,sentiment_2)
	kappa = getKappa(sentiment_1,sentiment_2)
	# Save agreement
	saveAgreement(N_AGREEMENT_SAMPLES, agreement, kappa)

# returns tweets on which the sentiment is agreed and their true sentiment
(test_tweets, true_sentiment) = getTestSet(samples, agreement)

if LOAD_MODEL==1:
	# load model (likelihood and priors)
	model = loadModel()
else:
	# get model (likelihood and priors)
	model = expectationMaximization(N_MODEL_STEPS)
	# save model (likelihood and priors)
	saveModel(model)

# get sentiment through naive bayes
sentiment = sentimentAnalysisNaiveBayes(test_tweets, model)

# resulting score between sentiment and true sentiment
print "Results Naive Bayes"
print macro_F1_score(sentiment, true_sentiment)

############

## do all tweets of a company and give
for FILENAME in FILENAMES: 
	print FILENAME
	tweets = getTweets(FILENAME)
	sentiment = sentimentAnalysisNaiveBayes(tweets, model)
	print "positive: "+str(sentiment.count(1))
	print "negative: "+str(sentiment.count(2))