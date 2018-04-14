#!/usr/bin/env python3
###################################
# CS B551 Fall 2017, Assignment #4
#
# Your names and user ids:
# Name: Yingnan Ju; ID: yiju
# Name: Yue Chen; ID: yc59
#
###################################
#
# The model files are:  nearest_model.txt
#                       adaboost_model.txt
#                       nnet_model.txt
#                       best_model.txt
# The output file is output.txt
#
# For train:
#   nearest:
#   ./orient.py train train-data.txt nearest_model.txt nearest
#   adaboost:
#   ./orient.py train train-data.txt adaboost_model.txt adaboost
#   nnet:
#   ./orient.py train train-data.txt nnet_model.txt nnet
#   best:
#   ./orient.py train train-data.txt best_model.txt best
#
# For test:
#   nearest:
#   ./orient.py test test-data.txt nearest_model.txt nearest
#   adaboost:
#   ./orient.py test test-data.txt adaboost_model.txt adaboost
#   nnet:
#   ./orient.py test test-data.txt nnet_model.txt nnet
#   best:
#   ./orient.py test test-data.txt best_model.txt best
#
#
#
# Put your report here!!
# Report by Yingnan Ju (yiju) and Yue Chen (yc59)
#
# Please see the report.pdf for a better formatted report with pictures.
# Please see the report.pdf for a better formatted report with pictures.
# Please see the report.pdf for a better formatted report with pictures.
#
#
# (1) a description of how you formulated the problem, including precisely defining the abstractions;
# We just used the data that the professor provided. The problem could be summarized to be predict the orientation of
# a picture through detect the color of (all or some) pixels of the thumbs. KNN, or nearest: regarding the color of
# each picture as a vector and compare the vector of the picture to be predicted with all vectors in the training set
# and find K vectors that are most similar with that vector (two vectors that have min distance between them)
# FORMULA: Please see the report.pdf for a better formatted report with pictures.
# Adaboost: or Adaptive Boosting. A boost classifier is a classifier in the form
# FORMULA: Please see the report.pdf for a better formatted report with pictures.
# Each ft(x) is a weak learner that takes an object x as input and returns a value indicating the class of the
# object [1]. In this case, it is a four-class problem and we get four classifiers eventually and each tells us if
# the picture is in this direction or not. If there are more positive answers or all answers are negative, we will
# compare the confidences of each classifier. The sign of the weak learner output identifies the predicted object
# class and the absolute value gives the confidence in that classification [1]. In this case, as the homework pdf
# said, we use simple decision stumps that simply compare the color of two different pixel. We will describe in the
# following part that how we chose the different pixels. Nnet: or neural network. We defined a three layers neural
# network. Input layer, hidden layer and output layer. The number of nodes in input layer is same with the input
# features.  The number of nodes in output layer is same with the number of output results. In this case we have
# four different answers so we defined four outputs, similar with the method in adaboost part. Similar, if there
# are more positive answers or all answers are negative, we will compare the confidences of each classifier. The
# number of nodes in the hidden layer was a parameter and in the following part we will describe how to decide it.
# The relation between input and output of each layer is:
# FORMULA: Please see the report.pdf for a better formatted report with pictures.
# This is the feed-forward process. We trained the network with backpropagation algorithm.
# FORMULA: Please see the report.pdf for a better formatted report with pictures.
# Best: same with KNN or nearest.
#
# (2) a brief description of how your program works;
# KNN, or nearest:
# Training part: simply copy all content train file into the model file (nearest_model.txt).
# Test part: for each picture in the test set, compare it with all vectors in the training set and find out which
# was the most common answer in the k nearest vectors.
# Adaboost: or Adaptive Boosting:
# Training part: take the color in index 172 (blue) and 3 (green) as a simple stump. Create four classifiers and
# each of them indicate one orientation of a picture. Repeat the iteration to train the classifiers in limited times
#  or the error rate is 0, which is almost impossible.
# Test part: for each picture in the test set we took the color in index 172 (blue) and 3 (green) and input them in
# each classifier and get the most possible answer as the prediction.
# Nnet: or neural network.:
# Training part: take the color in index 172 (blue) and 3 (green) as the input so the input layer of the network could
#  be simplified as two nodes. Create four networks and each of them indicate one orientation of a picture. Repeat the
#  iteration to train the networks in limited times with different parameters to find the best ones.
# Test part: for each picture in the test set we took the color in index 172 (blue) and 3 (green) and input them in
# each classifier and get the most possible answer as the prediction.
# Best: same with KNN or nearest.
#
# (3) a discussion of any problems, assumptions, simplifications, and/or design decisions you made; and
# Most classifiers or networks could only have two labels and the ones for multi labels are much more complicated. So,
# we use multi classifiers or networks to get the similar result. If there are more positive answers or all answers
# are negative, we will compare the confidences of each classifier or network.
#
# (4) answers to any questions asked below in the assignment.
# Please see the report.pdf for a better formatted report with pictures.
# Reference:
# [1] https://en.wikipedia.org/wiki/AdaBoost
# [2] http://blog.csdn.net/zjsghww/article/details/71485677
# [3] http://www.cnblogs.com/Finley/p/5946000.html
# [4] http://blog.csdn.net/miangangzhen/article/details/51281989
# [5] http://blog.csdn.net/ly_ysys629/article/details/72842067
# [6] https://ask.hellobi.com/blog/guodongwei1991/6709
#


import sys
import pickle
import heapq
import shutil
import time
from collections import Counter

from numpy import *


def read_file(path):
    file = open(path, 'r')
    photo_id_list = []
    photo_orientation_list = []
    photo_rgb_list = []
    for line in file:
        elements = line.split()
        photo_id_list.append(elements[0])
        photo_orientation_list.append(int(elements[1]))
        if len(elements) > 2:
            photo_rgb_list.append(array(elements[2:]).astype(int))
    return photo_id_list, photo_orientation_list, photo_rgb_list


def write_file(path, content_tuple):
    file = open(path, 'w')
    line_count = len(content_tuple[0])
    for i in range(line_count):
        for content in content_tuple:
            file.write(str(content[i]))
            file.write(' ')
        file.write('\n')
    file.close()
    print('results written to file "output.txt"')


def copy_file(file_path_1, file_path_2, length=-1):
    file1 = open(file_path_1, 'r')
    file2 = open(file_path_2, 'w')
    count = 0
    for line in file1:
        file2.write(line)
        count += 1
        if length != -1 and count >= length:
            break
    file1.close()
    file2.close()


def get_distance_between_vectors(vector1, vector2):
    if len(vector1) != len(vector2):
        return None

    temp_sum = 0
    for i in range(len(vector1)):
        difference = vector1[i] - vector2[i]
        temp_sum += difference * difference
    return math.sqrt(temp_sum)


def get_distance_between_vectors_simple(vector1, vector2):
    if len(vector1) != len(vector2):
        return None

    temp_sum = 0
    for i in range(len(vector1)):
        a, b = int(vector1[i]), int(vector2[i])
        temp_sum += a - b
    return temp_sum


def classify_stump(data_mat, dimension, threshold_val, threshold_inequal):
    result_array = ones((shape(data_mat)[0], 1))
    if threshold_inequal == 'lt':
        result_array[data_mat[:, dimension] <= threshold_val] = -1.0
    else:
        result_array[data_mat[:, dimension] > threshold_val] = -1.0
    return result_array


def build_stump(data_array, class_labels, d):
    data_matrix = mat(data_array)
    label_mat = mat(class_labels).T
    m, n = shape(data_matrix)
    num_steps = 10.0
    best_stumps = {}
    best_class_estimation = mat(zeros((m, 1)))
    min_error = inf
    for i in range(n):
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            threshold_val = range_min + float(j) * step_size
            for inequal in ['lt', 'gt']:
                predicted_vals = classify_stump(
                    data_matrix, i, threshold_val, inequal)
                err_array = mat(ones((m, 1)))
                err_array[predicted_vals == label_mat] = 0
                weighted_error = d.T * err_array
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_estimation = predicted_vals.copy()
                    best_stumps['dim'] = i
                    best_stumps['thresh'] = threshold_val
                    best_stumps['ineq'] = inequal
    return best_stumps, min_error, best_class_estimation


def adaboost_train_classifier(data_array, class_labels, iteration=40):
    weak_class_array = []
    m = shape(data_array)[0]
    d = mat(ones((m, 1)) / m)
    agg_class_estimation = mat(zeros((m, 1)))
    for i in range(iteration):
        best_stump, min_error, best_class_estimation = build_stump(data_array, class_labels, d)
        alpha = float(0.5 * log((1.0 - min_error) / max(min_error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_array.append(best_stump)
        exponent = multiply(-1 * alpha * mat(class_labels).T, best_class_estimation)
        d = multiply(d, exp(exponent))
        d = d / d.sum()
        agg_class_estimation += alpha * best_class_estimation
        agg_errors = multiply(sign(agg_class_estimation) != mat(class_labels).T, ones((m, 1)))
        errors_rate = agg_errors.sum() / m
        if errors_rate == 0.0:
            break
    return weak_class_array


def classify_data(data_to_class, classifier):
    data_matrix = mat(data_to_class)
    m = shape(data_matrix)[0]
    agg_class_estimation = mat(zeros((m, 1)))
    for i in range(len(classifier)):
        class_estimation = classify_stump(data_matrix, classifier[i]['dim'],
                                          classifier[i]['thresh'],
                                          classifier[i]['ineq']
                                          )
        agg_class_estimation += classifier[i]['alpha'] * class_estimation
    return agg_class_estimation


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    result = []
    for i in range(m):
        result.append([fill] * n)
    return result


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# reference:
# http://blog.csdn.net/zjsghww/article/details/71485677
# http://www.cnblogs.com/Finley/p/5946000.html
# http://blog.csdn.net/miangangzhen/article/details/51281989
#
class NeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []

    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                # self.input_weights[i][h] = rand(-0.2, 0.2)
                self.input_weights[i][h] = 0.1
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                # self.output_weights[h][o] = rand(-2.0, 2.0)
                self.output_weights[h][o] = 1
        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # feed forward
        self.predict(case)
        # get output layer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)

    def test(self, right_label):
        train_path = "train-data.txt"
        test_path = "test-data.txt"
        cases, labels = self.load_simple_data(right_label, train_path)
        test_cases, test_labels = self.load_simple_data(right_label, test_path)

        self.setup(2, 7, 1)
        t = 3000
        self.train(cases[:t], labels[:t], 80, 0.005, 0.05)
        result_list = []
        for case in test_cases[:]:
            # print(self.predict(case))
            result_list.append(self.predict(case))
        return result_list

    @staticmethod
    def read_file(path):
        file = open(path, 'r')
        photo_id_list = []
        photo_orientation_list = []
        photo_rgb_list = []
        for line in file:
            elements = line.split()
            photo_id_list.append(elements[0])
            photo_orientation_list.append(int(elements[1]))
            int_list = [int(elements[172]), int(elements[3])]
            # for ele in elements[2:]:
            #     int_list.append(int(ele))
            if len(elements) > 2:
                photo_rgb_list.append(int_list)
        return photo_id_list, photo_orientation_list, photo_rgb_list

    def load_simple_data(self, right_label, path):
        train_id_list, train_orientation_list, train_rgb_list = self.read_file(path)
        dat_mat = train_rgb_list[:]
        temp_list = [[1] if label == right_label else [0] for label in train_orientation_list]
        return dat_mat, temp_list


def train_nearest(input_file_path, model_file_path):
    # id_list, orientation_list, rgb_list = read_file(input_file_path)
    shutil.copy2(input_file_path, model_file_path)
    # copy_file(input_file_path, model_file_path, 4000)


# reference
# https://ask.hellobi.com/blog/guodongwei1991/6709
# http://blog.csdn.net/ly_ysys629/article/details/72842067
#
def train_adaboost(input_file_path, model_file_path):
    train_id_list, train_orientation_list, train_rgb_list = read_file(input_file_path)
    index1, index2 = 172 - 2, 3 - 2
    extracted_rgb_list = []
    classifier_list = []
    for line in train_rgb_list:
        extracted_rgb_list.append([line[index1], line[index2]])
    for i in range(4):
        signed_orientation_list = [1 if label == 90 * i else -1 for label in train_orientation_list]
        classifier = adaboost_train_classifier(extracted_rgb_list, matrix(signed_orientation_list), iteration=41)
        classifier_list.append(classifier)
    pickle.dump(classifier_list, open(model_file_path, 'wb'))


# reference:
# http://blog.csdn.net/zjsghww/article/details/71485677
# http://www.cnblogs.com/Finley/p/5946000.html
# http://blog.csdn.net/miangangzhen/article/details/51281989
#
def train_nnet(input_file_path, model_file_path):
    nn_list = [NeuralNetwork(), NeuralNetwork(), NeuralNetwork(), NeuralNetwork()]
    for i in range(len(nn_list)):
        cases, labels = nn_list[i].load_simple_data(i * 90, input_file_path)
        nn_list[i].setup(2, 7, 1)
        nn_list[i].train(cases[:3000], labels[:3000], 80, 0.005, 0.05)
    pickle.dump(nn_list, open(model_file_path, 'wb'))


def test_nearest(input_file_path, model_file_path):
    # start = time.time()
    k = 11
    train_id_list, train_orientation_list, train_rgb_list = read_file(model_file_path)
    test_id_list, test_orientation_list, test_rgb_list = read_file(input_file_path)
    result_list = []

    # test
    # file = open("output", 'w')
    correct_count = 0

    # for i in range(20):
    for i in range(len(test_id_list)):
        distance_list = []
        for j in range(len(train_id_list)):
            distance_list.append(linalg.norm(test_rgb_list[i] - train_rgb_list[j]))
            # distance_list.append(get_distance_between_vectors(test_rgb_list[i], train_rgb_list[j]))
        largest_k_list = heapq.nsmallest(k, distance_list)
        label_list = []
        for value in largest_k_list:
            index = distance_list.index(value)
            label_list.append(train_orientation_list[index])
        counter = Counter(label_list)
        # print(counter.most_common(1))
        result_list.append(counter.most_common(1)[0][0])

        # test
        # file.write(test_id_list[i])
        # file.write(' ')
        # file.write(result_list[i])
        # file.write('\n')
        correct_count += (1 if result_list[i] == test_orientation_list[i] else 0)
    # print(k)
    # print(time.time()-start)

    write_file("output.txt", (test_id_list, result_list))

    print("accuracy: ", correct_count / len(test_id_list))


def test_adaboost(input_file_path, model_file_path):
    test_id_list, test_orientation_list, test_rgb_list = read_file(input_file_path)
    index1, index2 = 172 - 2, 3 - 2
    classifier_list = pickle.load(open(model_file_path, 'rb'))
    extracted_rgb_list = []
    for line in test_rgb_list:
        extracted_rgb_list.append([line[index1], line[index2]])
    total_count = len(extracted_rgb_list)
    correct_count = 0
    result_list = []
    for i in range(total_count):
        possibility_list = []
        for classifier in classifier_list:
            result = classify_data(extracted_rgb_list[i], classifier)
            possibility_list.append(result)
        result = 90 * possibility_list.index(max(possibility_list))
        possibility_list.append(result)
        result_list.append(result)
        # print(result)
        right_answer = test_orientation_list[i]
        if result == right_answer:
            correct_count += 1
        # print(correct_count / total_count)
    write_file("output.txt", (test_id_list, result_list))
    print("accuracy: ", correct_count / total_count)


def test_nnet(input_file_path, model_file_path):
    test_id_list, test_orientation_list, test_rgb_list = read_file(input_file_path)
    nn_list = pickle.load(open(model_file_path, 'rb'))

    possibility_list = []
    for i in range(len(nn_list)):
        test_cases, test_labels = nn_list[i].load_simple_data(0 * 90, input_file_path)
        result_list = []
        for case in test_cases[:]:
            # print(self.predict(case))
            result_list.append(nn_list[i].predict(case))
        possibility_list.append(result_list)
    pred_list = []
    for i in range(len(possibility_list[0])):
        temp_list = [l[i] for l in possibility_list]
        pred_list.append(90 * temp_list.index(max(temp_list)))

    write_file("output.txt", (test_id_list, pred_list))
    correct_answer = nn_list[0].read_file("test-data.txt")[1]
    all_count = len(pred_list)
    correct_count = 0
    for i in range(all_count):
        if pred_list[i] == correct_answer[i]:
            correct_count += 1
    print("accuracy: ", correct_count / all_count)


def select_train_method(input_method, input_file, model_path):
    if input_method == 'nearest':
        train_nearest(input_file, model_path)
    elif input_method == 'adaboost':
        train_adaboost(input_file, model_path)
    elif input_method == 'nnet':
        train_nnet(input_file, model_path)
    elif input_method == 'best':
        train_nearest(input_file, model_path)
    else:
        sys.exit("wrong method: should be 'nearest', 'adaboost', 'nnet', or 'best'")


def select_test_method(input_method, input_file, model_file):
    if input_method == 'nearest':
        test_nearest(input_file, model_file)
    elif input_method == 'adaboost':
        test_adaboost(input_file, model_file)
    elif input_method == 'nnet':
        test_nnet(input_file, model_file)
    elif input_method == 'best':
        test_nearest(input_file, model_file)
    else:
        sys.exit("wrong method: should be 'nearest', 'adaboost', 'nnet', or 'best'")


if sys.version_info[0] < 3:
    print("Must be using Python 3")
    sys.stderr.write("Error: Your Python interpreter must be 3.0 or higher\n")
    sys.exit(-1)

if len(sys.argv) < 5:
    sys.exit("No enough parameters. Please check the input format:\n"
             "./orient.py train train_file.txt model_file.txt [model]\n"
             "or\n"
             "./orient.py test test_file.txt model_file.txt [model]")

# start = time.time()
ml_function = sys.argv[1]
input_path = sys.argv[2]
model_path = sys.argv[3]
method = sys.argv[4]
# print(function, input_file_path, model_file_path, method)

if ml_function == 'train':
    select_train_method(method, input_path, model_path)
elif ml_function == 'test':
    select_test_method(method, input_path, model_path)
else:
    sys.exit("wrong function: should be 'train' or 'test'")
# print(time.time() - start)
