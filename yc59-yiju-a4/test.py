# from __future__ import division
from numpy import *


class AdaBoost:
    def __init__(self, training_set):
        self.training_set = training_set
        self.N = len(self.training_set)
        self.weights = ones(self.N) / self.N
        self.RULES = []
        self.ALPHA = []

    def set_rule(self, func, test=False):
        errors = array([t[1] != func(t[0]) for t in self.training_set])
        e = (errors * self.weights).sum()
        if test: return e
        alpha = 0.5 * log((1 - e) / e)
        print('e=%.2f a=%.2f' % (e, alpha))
        w = zeros(self.N)
        for i in range(self.N):
            if errors[i] == 1:
                w[i] = self.weights[i] * exp(alpha)
            else:
                w[i] = self.weights[i] * exp(-alpha)
        self.weights = w / w.sum()
        self.RULES.append(func)
        self.ALPHA.append(alpha)

    def evaluate(self):
        NR = len(self.RULES)
        for (x, l) in self.training_set:
            hx = [self.ALPHA[i] * self.RULES[i](x) for i in range(NR)]
            print(x, sign(l) == sign(sum(hx)))


examples = []
examples.append(((1, 2), 1))
examples.append(((1, 4), 1))
examples.append(((2.5, 5.5), 1))
examples.append(((3.5, 6.5), 1))
examples.append(((4, 5.4), 1))
examples.append(((2, 1), -1))
examples.append(((2, 4), -1))
examples.append(((3.5, 3.5), -1))
examples.append(((5, 2), -1))
examples.append(((5, 5.5), -1))
for x1 in range(4, 194, 12):
    for x2 in range(3, 193, 12):
        adaboosting = Adaboosting()
        D = mat(ones((5, 1)) / 5)
        classifierArr = []
        for i in range(4):
            dataMat, lableMat = adaboosting.load_simple_data(90 * i, x1, x2)
            classifier = adaboosting.adaBoostingDs(dataMat, lableMat, 15)
            classifierArr.append(classifier)

m = AdaBoost(examples)
m.set_rule(lambda x: 2 * (x[0] < 1.5) - 1)
m.set_rule(lambda x: 2 * (x[0] < 4.5) - 1)
m.set_rule(lambda x: 2 * (x[1] > 5) - 1)
m.evaluate()
