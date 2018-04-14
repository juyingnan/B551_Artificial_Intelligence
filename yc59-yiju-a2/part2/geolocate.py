#!/usr/bin/env python3

# Professor David Crandall
# B551 Elements of Artificial Intelligence
# Yue Chen and Yingnan Ju
# October 22nd, 2017

# Assignment 2
# Part 2 Tweet Classification

import sys
import string
import math

n_word = 5
v_shreshold = 100
common_shreshold = 3000
progress_bar = 0


# Read in the training tweets, clean up

def read_in(file):
    geolocation = []
    rawTweets = []
    lines = file.readlines()
    translator = str.maketrans(dict.fromkeys(string.punctuation))

    for line in lines:
        eachTweet = line.strip()
        eachTweet = str(eachTweet, "utf-8", "ignore").lower().split()
        if len(eachTweet) > 1:
            if ',_' in eachTweet[0]:
                geolocation.append(eachTweet[0])
                rawTweets.append(" ".join(eachTweet[1:]))
            else:
                rawTweets[-1] += ' '
                rawTweets[-1] += ' '.join(eachTweet)
            rawTweets[-1] = rawTweets[-1].translate(translator)

    geolocationSet = list(set(geolocation))

    return ((geolocation, rawTweets, geolocationSet))


# input: geolocation: a list of string
# input: rawTweets: a list of string
# input: geolocationSet: a unique list of string
# geolocation and rawTweets are 1-1 mapped.
# output: a list of lists
# each sublist a group of tweets whose geolocation are the same.
# the order of the groups should be the same as the order in geolocationSet.
def group_tweets(geolocation, rawTweets, geolocationSet):
    result_list = []
    for i in range(0, len(geolocationSet)):
        result_list.append([])
    for i in range(0, len(geolocation)):
        index = geolocationSet.index(geolocation[i])
        result_list[index].append(rawTweets[i])
    return result_list


# test group_tweets
# g = ['city0', 'city1', 'city2', 'city3', 'city3', 'city2', 'city1', 'city1', ]
# t = ['t0', 't1', 't2', 't3', 't3', 't2', 't1', 't1', ]
# gs = ['city0', 'city1', 'city2', 'city3', ]
# rl = group_tweets(g, t, gs)
# print(rl)

# Generate feature vector
def get_feature_vector_ngram(rawTweets, n):
    featureVector = []
    featureVector_count_list = []
    for eachTweet in rawTweets:
        eachFeatureVector = [eachTweet[i:i + n] for i in range(len(eachTweet) - n + 1)]
        for eachElement in eachFeatureVector:
            if eachElement not in featureVector:
                featureVector.append(eachElement)
                featureVector_count_list.append(1)
            else:
                featureVector_count_list[featureVector.index(eachElement)] += 1
    for item in featureVector[::-1]:
        if featureVector_count_list[featureVector.index(item)] < v_shreshold:
            featureVector_count_list.pop(featureVector.index(item))
            featureVector.pop(featureVector.index(item))
    return featureVector


def get_feature_vector_nword(rawTweets, n):
    featureVector = []
    featureVector_count_list = []

    for eachTweet in rawTweets:
        tweet_word_list = eachTweet.split()
        eachFeatureVector = [tweet_word_list[i:i + n] for i in range(len(tweet_word_list) - n + 1)]
        for eachElement in eachFeatureVector:
            if eachElement not in featureVector:
                featureVector.append(''.join(eachElement))
                featureVector_count_list.append(1)
            else:
                featureVector_count_list[featureVector.index(eachElement)] += 1
    for item in featureVector[::-1]:
        if featureVector_count_list[featureVector.index(item)] < v_shreshold:
            featureVector_count_list.pop(featureVector.index(item))
            featureVector.pop(featureVector.index(item))
    return featureVector


def get_feature_vector(rawTweets):
    featureVector = []
    featureVector_count_list = []

    for eachTweet in rawTweets:
        eachFeatureVector = eachTweet.split()
        for eachElement in eachFeatureVector:
            if eachElement not in featureVector:
                featureVector.append(eachElement)
                featureVector_count_list.append(1)
            else:
                featureVector_count_list[featureVector.index(eachElement)] += 1
    for item in featureVector[::-1]:
        if featureVector_count_list[featureVector.index(item)] < v_shreshold or \
                        featureVector_count_list[featureVector.index(item)] > common_shreshold:
            featureVector_count_list.pop(featureVector.index(item))
            featureVector.pop(featureVector.index(item))
    return (featureVector)


def calculate_class_probability(numberOfTweets, groupedTweets):
    classProbability = []
    for eachGroup in groupedTweets:
        classProbability.append(len(eachGroup) / numberOfTweets)

    return (classProbability)


# return n by m list of lists
# n = number of features
# m = number of classes
# return P(Xi|Ci) for each element in the featureVector
def calculate_log_likelihood(groupedTweets, featureVector, geolocationSet):
    result_matrix = []
    for i in range(len(geolocationSet)):
        result_matrix.append([])
        for j in range(len(featureVector)):
            result_matrix[i].append(0)
    for word in featureVector:
        group_count_list = []
        for i in range(len(geolocationSet)):
            group_count = 0
            for tweet in groupedTweets[i]:
                count = tweet.count(word)
                group_count += count
            if group_count < 1:
                group_count = 1
            group_count_list.append(group_count)
        all_count = sum(group_count_list)
        for i in range(len(geolocationSet)):
            result_matrix[i][featureVector.index(word)] = (
                1.0 * group_count_list[i] / all_count) if all_count != 0 else 0

    # print top 5 words for 12 cities
    print('Top 5 words:')
    for i in range(len(geolocationSet)):
        top_5_list = []
        temp_list = list(result_matrix[i])
        for j in range(0, 5):
            top_5_list.append(max(temp_list))
            temp_list.remove(max(temp_list))
        word_list = []
        for top in top_5_list:
            k = result_matrix[i].index(top)
            word_list.append(featureVector[k])
        print(geolocationSet[i], ':', ' '.join(word_list))
    return result_matrix


def map_estimation(rawTweets, log_likelihood_matrix, train_class_probability,
                   feature_vector):
    result_matrix = []
    for i in range(len(rawTweets)):
        result_matrix.append([])
        for j in range(len(train_class_probability)):
            result_matrix[i].append(0)

    count = 0
    count_in_list = 0
    for i in range(len(rawTweets)):
        for j in range(len(train_class_probability)):
            p = 0
            for word in rawTweets[i].split():
                count += 1
                if word in feature_vector:
                    count_in_list += 1
                    p += math.log(log_likelihood_matrix[j][feature_vector.index(word)], math.e)
            p *= train_class_probability[j]
            # (1.0 / len(rawTweets[i].split())
            result_matrix[i][j] = p

    # print(count, count_in_list)
    return result_matrix


def map_estimation_nword(rawTweets, log_likelihood_matrix, train_class_probability,
                         feature_vector, n):
    result_matrix = []
    for i in range(len(rawTweets)):
        result_matrix.append([])
        for j in range(len(train_class_probability)):
            result_matrix[i].append(0)

    for i in range(len(rawTweets)):
        for j in range(len(train_class_probability)):
            p = 0
            tweet_word_list = rawTweets[i].split()

            # This is adapted from a previous code of mine for SemEval
            eachFeatureVector = [tweet_word_list[i:i + n] for i in range(len(tweet_word_list) - n + 1)]
            for eachElement in eachFeatureVector:
                if eachElement in feature_vector:
                    p += math.log(log_likelihood_matrix[j][feature_vector.index(eachElement)], math.e)
            p *= train_class_probability[j]
            # (1.0 / len(rawTweets[i].split())
            result_matrix[i][j] = p

    return result_matrix


def map_estimation_ngram(rawTweets, log_likelihood_matrix, train_class_probability,
                         feature_vector, n):
    result_matrix = []
    for i in range(len(rawTweets)):
        result_matrix.append([])
        for j in range(len(train_class_probability)):
            result_matrix[i].append(0)
    count = 0
    count_in_list = 0
    for i in range(len(rawTweets)):
        for j in range(len(train_class_probability)):
            p = 0
            eachFeatureVector = [rawTweets[i][i:i + n] for i in range(len(rawTweets[i]) - n + 1)]
            for eachElement in eachFeatureVector:
                count += 1
                if eachElement in feature_vector:
                    count_in_list += 1
                    p += math.log(log_likelihood_matrix[j][feature_vector.index(eachElement)], math.e)
            p *= train_class_probability[j]
            # (1.0 / len(rawTweets[i].split())
            result_matrix[i][j] = p
    print(count, count_in_list)

    return result_matrix


# # calculate_log_likelihood test code
# gs = ['city0', 'city1', 'city2', 'city3', ]
# fv = ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', ]
# gt = [['w0 w1 w0', 'w2 w0 w0', 'w0 w0 w0'],
#       ['w0 w1 w0', 'w2 w0 w0', 'w0 w0 w0'],
#       ['w0 w1 w0', 'w2 w3 w4', 'w4 w5 w6'],
#       ['w3 w3 w0', 'w1 w0 w2', 'w6 w6 w6']]
# result = calculate_log_likelihood(gt, fv, gs)
# for line in result:
#     print(line)
# # map_estimation test code
# rt = ['w0 w1 w0',
#       'w2 w0 w0',
#       'w0 w0 w0',
#       'w2 w3 w4',
#       'w6 w6 w6',
#       'w4 w5 w6']
# llm = result
# tcp = [0.5, 0.2, 0.2, 0.1]
# result = map_estimation(rt, llm, tcp, fv)
# for line in result:
#     print(line)

# Parse input parameters

trainFileName = sys.argv[1]
testFileName = sys.argv[2]
outputFileName = sys.argv[3]

trainFile = open(trainFileName, "rb")
testFile = open(testFileName, "rb")
outputFile = open(outputFileName, "w")

# Main

trainGeolocation, trainRawTweets, trainGeolocationSet = read_in(trainFile)
# print('5%')
common_shreshold = len(trainRawTweets) / 15
v_shreshold = math.sqrt(len(trainRawTweets)) / 3
testGeolocation, testRawTweets, testGeolocationSet = read_in(testFile)
# print('10%')
numberOfTrainTweets = len(trainRawTweets)
# print('11%')
groupedTweets = group_tweets(trainGeolocation, trainRawTweets, trainGeolocationSet)
# print('20%')
# trainFeatureVector = get_feature_vector_nword(trainRawTweets, n_word)
# trainFeatureVector = get_feature_vector_ngram(trainRawTweets, n_word)
trainFeatureVector = get_feature_vector(trainRawTweets)
# print(len(trainFeatureVector))
# print('45%')
trainClassProbability = calculate_class_probability(numberOfTrainTweets, groupedTweets)
# print('55%')
trainLogLikelihood = calculate_log_likelihood(groupedTweets, trainFeatureVector, trainGeolocationSet)
# print('75%')
testMapEstimation = map_estimation(testRawTweets, trainLogLikelihood, trainClassProbability, trainFeatureVector)
# print('95%')
guess_city_list = []
for line in testMapEstimation:
    # print(line)
    index = line.index(max(line))
    guess_city_list.append(trainGeolocationSet[index])
count = 0
for i in range(len(guess_city_list)):
    if guess_city_list[i] == testGeolocation[i]:
        count += 1
print(1.0 * count / len(testGeolocation))

fo = open(outputFileName, "w")
for i in range(len(guess_city_list)):
    s = guess_city_list[i] + ' ' + testGeolocation[i] + ' ' + testRawTweets[i] + '\n'
    fo.write(s)
fo.close
# print("tuzi")
# print('100%')
