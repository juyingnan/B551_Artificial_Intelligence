# -*- coding:utf-8 -*-
from numpy import *


class Adaboosting(object):
    def read_file(self, path, x1, x2):
        file = open(path, 'r')
        photo_id_list = []
        photo_orientation_list = []
        photo_rgb_list = []
        for line in file:
            elements = line.split()
            photo_id_list.append(elements[0])
            photo_orientation_list.append(int(elements[1]))
            if len(elements) > 2:
                photo_rgb_list.append(array([elements[x1], elements[x2]]).astype(int))
                # photo_rgb_list.append(array(elements[2:]).astype(int))
        return photo_id_list, photo_orientation_list, photo_rgb_list

    def loadSimpData(self, right_label, x1, x2):
        # datMat = matrix(
        #     [[1., 2.1, 2],
        #      [2., 1.1, 2],
        #      [1.3, 1., 1],
        #      [1., 1., 5],
        #      [2., 1., 2]])
        # classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
        train_id_list, train_orientation_list, train_rgb_list = self.read_file("train-data.txt", x1, x2)
        datMat = train_rgb_list[:]
        temp_list = [1 if label == right_label else -1 for label in train_orientation_list]
        classLabels = matrix(temp_list[:])
        return datMat, classLabels

    def stumpClassify(self, datMat, dimen, threshVal, threshIneq):
        # print "-----data-----"
        # print datMat
        retArr = ones((shape(datMat)[0], 1))
        if threshIneq == 'lt':
            retArr[datMat[:, dimen] <= threshVal] = -1.0  # 小于阈值的列都为-1
        else:
            retArr[datMat[:, dimen] > threshVal] = -1.0  # 大于阈值的列都为-1
        # print "---------retArr------------"
        # print retArr
        return retArr

    def buildStump(self, dataArr, classLables, D):
        """
        单层决策树生成函数
        """
        dataMatrix = mat(dataArr)
        lableMat = mat(classLables).T
        m, n = shape(dataMatrix)
        numSteps = 10.0  # 步数，影响的是迭代次数，步长
        bestStump = {}  # 存储分类器的信息
        bestClassEst = mat(zeros((m, 1)))  # 最好的分类器
        minError = inf  # 迭代寻找最小错误率
        for i in range(n):
            # 求出每一列数据的最大最小值计算步长
            rangeMin = dataMatrix[:, i].min()
            rangeMax = dataMatrix[:, i].max()
            stepSize = (rangeMax - rangeMin) / numSteps
            # j唯一的作用用步数去生成阈值，从最小值大最大值都与数据比较一边了一遍
            for j in range(-1, int(numSteps) + 1):
                threshVal = rangeMin + float(j) * stepSize  # 阈值
                for inequal in ['lt', 'gt']:
                    predictedVals = self.stumpClassify(
                        dataMatrix, i, threshVal, inequal)
                    errArr = mat(ones((m, 1)))
                    errArr[predictedVals == lableMat] = 0  # 为1的 表示i分错的
                    weightedError = D.T * errArr  # 分错的个数*权重(开始权重=1/M行)
                    # print "split: dim %d, thresh %.2f, thresh ineqal:\
                    # %s,the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                    if weightedError < minError:  # 寻找最小的加权错误率然后保存当前的信息
                        minError = weightedError
                        bestClassEst = predictedVals.copy()  # 分类结果
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshVal
                        bestStump['ineq'] = inequal
        # print bestStump
        # print minError
        # print bestClassEst  # 类别估计
        return bestStump, minError, bestClassEst

    def adaBoostingDs(self, dataArr, classLables, numIt=40):
        """
        基于单层决策树的AdaBoosting训练过程：
        """
        weakClassArr = []  # 最佳决策树数组
        m = shape(dataArr)[0]
        D = mat(ones((m, 1)) / m)
        aggClassEst = mat(zeros((m, 1)))
        for i in range(numIt):
            bestStump, minError, bestClassEst = self.buildStump(
                dataArr, classLables, D)
            # print("bestStump:", bestStump)
            # print("D:", D.T)
            alpha = float(
                0.5 * log((1.0 - minError) / max(minError, 1e-16)))
            bestStump['alpha'] = alpha
            weakClassArr.append(bestStump)
            # print("alpha:", alpha)
            # print("classEst:", bestClassEst.T)  # 类别估计

            expon = multiply(-1 * alpha * mat(classLables).T, bestClassEst)
            D = multiply(D, exp(expon))
            D = D / D.sum()

            aggClassEst += alpha * bestClassEst
            # print("aggClassEst ；", aggClassEst.T)
            # 累加错误率
            aggErrors = multiply(sign(aggClassEst) !=
                                 mat(classLables).T, ones((m, 1)))
            # 错误率平均值
            errorsRate = aggErrors.sum() / m
            # print("total error:", errorsRate, "\n")
            if errorsRate == 0.0:
                break
        # print("weakClassArr:", weakClassArr)
        return weakClassArr

    def adClassify(self, datToClass, classifierArr):
        """
        预测分类：
        datToClass：待分类数据
        classifierArr: 训练好的分类器数组
        """
        dataMatrix = mat(datToClass)
        m = shape(dataMatrix)[0]
        aggClassEst = mat(zeros((m, 1)))
        # print()
        for i in range(len(classifierArr)):  # 有多少个分类器迭代多少次
            # 调用第一个分类器进行分类
            classEst = self.stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                          classifierArr[i]['thresh'],
                                          classifierArr[i]['ineq']
                                          )
            # alpha 表示每个分类器的权重，
            # print(classEst)
            aggClassEst += classifierArr[i]['alpha'] * classEst
            # print(aggClassEst)
        # return sign(aggClassEst)
        return aggClassEst


if __name__ == "__main__":
    accuracy_list = []
    co_list = []

    adaboosting_temp = Adaboosting()
    dataMat_list = []
    lableMat_list=[]
    for i in range(4):
        dataMat, lableMat = adaboosting_temp.loadSimpData(90 * i, x1=172, x2=3)
        dataMat_list.append(dataMat)
        lableMat_list.append(lableMat)


    adaboosting = Adaboosting()
    classifierArr = []
    for i in range(4):
        classifier = adaboosting.adaBoostingDs(dataMat_list[i], lableMat_list[i], iter)
        classifierArr.append(classifier)

    # 预测数据
    # result = adaboosting.adClassify([2, 1, 2], classifierArr)
    file = open("test-data.txt", 'r')
    photo_id_list = []
    photo_orientation_list = []
    photo_rgb_list = []
    for line in file:
        elements = line.split()
        photo_id_list.append(elements[0])
        photo_orientation_list.append(int(elements[1]))
        if len(elements) > 2:
            photo_rgb_list.append(array([elements[172], elements[3]]).astype(int))
            # photo_rgb_list.append(array(elements[2:]).astype(int))

    total_count = len(photo_rgb_list)
    correct_count = 0
    for i in range(total_count):
        result_list = []
        for classifier in classifierArr:
            result = adaboosting.adClassify(photo_rgb_list[i], classifier)
            result_list.append(result)
        result = 90 * result_list.index(max(result_list))
        # print(result)
        right_answer = photo_orientation_list[i]
        if result == right_answer:
            correct_count += 1
    # print(correct_count / total_count)
    accuracy_list.append(correct_count / total_count)
    co_list.append(iter)
    print(correct_count / total_count, iter)

    print(max(accuracy_list))
    print(co_list[accuracy_list.index(max(accuracy_list))])
