import math
import random
from numpy import *


# random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class BPNeuralNetwork:
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
                self.input_weights[i][h] = rand(-0.2, 0.2)
                #self.input_weights[i][h] = 0.1
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
                #self.output_weights[h][o] = 1.0
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

    def test(self, right_label, parameter):
        dataMat_list = []
        lableMat_list = []
        # for i in range(4):
        #     dataMat, lableMat = self.loadSimpData(90 * i, x1=172, x2=3)
        #     dataMat_list.append(dataMat)
        #     lableMat_list.append(lableMat)
        train_path = "train-data.txt"
        test_path = "test-data.txt"
        cases, labels = self.loadSimpData(right_label, train_path)
        test_cases, test_labels = self.loadSimpData(right_label, test_path)

        self.setup(2, 7, 1)
        t = 3000
        self.train(cases[:t], labels[:t], 80, 0.005, 0.05)
        result_list = []
        for case in test_cases[:]:
            # print(self.predict(case))
            result_list.append(self.predict(case))
        return result_list

    def read_file(self, path):
        file = open(path, 'r')
        photo_id_list = []
        photo_orientation_list = []
        photo_rgb_list = []
        for line in file:
            elements = line.split()
            photo_id_list.append(elements[0])
            photo_orientation_list.append(int(elements[1]))
            int_list = []
            # for ele in elements[2:]:
            #     int_list.append(int(ele))
            int_list.append(int(elements[172]))
            int_list.append(int(elements[3]))
            if len(elements) > 2:
                photo_rgb_list.append(int_list)
        return photo_id_list, photo_orientation_list, photo_rgb_list

    def loadSimpData(self, right_label, path):
        # datMat = matrix(
        #     [[1., 2.1, 2],
        #      [2., 1.1, 2],
        #      [1.3, 1., 1],
        #      [1., 1., 5],
        #      [2., 1., 2]])
        # classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
        train_id_list, train_orientation_list, train_rgb_list = self.read_file(path)
        datMat = train_rgb_list[:]
        temp_list = [[1] if label == right_label else [0] for label in train_orientation_list]
        # classLabels = list(temp_list[:])
        return datMat, temp_list


if __name__ == '__main__':
    for j in range(0, 100,1):
        random.seed(j)
        nn_list = []
        nn1 = BPNeuralNetwork()
        nn2 = BPNeuralNetwork()
        nn3 = BPNeuralNetwork()
        nn4 = BPNeuralNetwork()
        nn_list.append(nn1)
        nn_list.append(nn2)
        nn_list.append(nn3)
        nn_list.append(nn4)
        possibility_list = []
        for i in range(4):
            possibility_list.append(nn_list[i].test(i * 90, j))
        pred_list = []
        for i in range(len(possibility_list[0])):
            temp_list = [l[i] for l in possibility_list]
            pred_list.append(90 * temp_list.index(max(temp_list)))
        print(pred_list)

        correct_answer = nn1.read_file("test-data.txt")[1]
        all_count = len(pred_list)
        correct_count = 0
        for i in range(all_count):
            if pred_list[i] == correct_answer[i]:
                correct_count += 1
        print(correct_count / all_count, j)
