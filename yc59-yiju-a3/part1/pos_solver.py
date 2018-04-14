###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
# Name: Yingnan Ju; ID: yiju
# Name: Yue Chen; ID: yc59
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
# Report by Yingnan Ju (yiju) and Yue Chen (yc59)
#
# Final test result of bc.test file:
#
# ==> So far scored 2000 sentences with 29442 words.
#                   Words correct:     Sentences correct:
#   0. Ground truth:      100.00%              100.00%
#     1. Simplified:       93.95%               47.50%
#         2. HMM VE:       94.45%               50.95%
#        3. HMM MAP:       94.92%               53.60%
# ----

# (1) a description of how you formulated the problem,
# including precisely defining the abstractions (e.g. HMM formulation); #

#      a. simplified: p(TAG i = tag i) = arg max p(TAG i = tag i | WORD i = word i)
#           Just find the max probability of p(TAG i = tag i | WORD i = word i)
#
#      b. HMM VE: p(TAG i = tag i) = arg max p(TAG i = tag i | WORD i = word i)
#          = arg max p(TAG i = tag i, WORD i = word i) / p(WORD i = word i)
#          = arg max p(WORD i = word i | TAG i = tag i) p(TAG i = tag i) / p(WORD i = word i)
#          = arg max p(WORD i = word i | TAG i = tag i) p(TAG i | TAG i-1) p(TAG i-1 = tag i-1) / p(WORD i = word i)
#          o= arg max p(WORD i = word i | TAG i = tag i) p(TAG i | TAG i-1)
#          because we don't need to calculate the exact probability but only need to find the highest probability
#           (o= means "proportional to")
#
#      c. HMM MAP: p(tag0, tag1, ..., tag n) = arg max p(tag0, tag1, ..., tag n | word0, word1, ..., word n)
#         = arg max p(word0, word1, ..., word n | tag0, tag1, ..., tag n)
#             * p(tag0, tag1, ..., tag n) / p(word0, word1, ..., word n)
#          o= arg max p(word0, word1, ..., word n | tag0, tag1, ..., tag n)  * p(tag0, tag1, ..., tag n)
#          = arg max p(tag0, tag1, ..., tag n) * PRODUCT: p(WORD i = word i | TAG i = tag i)
#           (o= means "proportional to")
#
#       d. logarithm of the posterior probability:
#           p(WORD i = word i | TAG i = tag i)
#           = p(TAG i = tag i | WORD i = word i) * P(WORD i = word i) / p(TAG i = tag i)
#           then, Log(p(WORD i = word i | TAG i = tag i))
#

# (2) a brief description of how your program works;

#      in training part, get:
#          p(TAG i = tag i | WORD i = word i)  ,a matrix:
#
#                   |   tag 0   tag 1   ...     tag n
#           --------|--------------------------------------
#            word 0 | p(t0|w0)  p(t1|w0) ...
#            word 1 |
#            ...    |
#            word n |                    ...    p(tn|wn)
#
#          p(WORD i = word i | TAG i = tag i)  ,a matrix:
#
#                   |  word 0   word 1   ...    word n
#           --------|--------------------------------------
#            tag 0  | p(w0|t0)  p(w1|t0) ...
#            tag 1  |
#            ...    |
#            tag n  |                    ...    p(wn|tn)
#
#          p(TAG i | TAG i-1)  ,a matrix:
#
#                   |   tag 0   tag 1   ...     tag n       end (this part has not been used yet)
#           --------|------------------------------------------------
#            tag 0  | p(t0|t0)  p(t1|t0) ...                p(end|t0)
#            tag 1  |
#            ...    |                                       ...
#            tag n  |                    ...    p(tn|tn)    p(end|tn)
#            start  | p(t0|start)        ...    p(tn|start) p(end|start)
#
#          p(TAG i = tag i), a frequency list:
#           [p(tag 0), p(tag 1), ... , p(tag n)]
#
#          p(WORD i = word i), a frequency list
#           [p(word 0), p(word 1), ... , p(word n)]
#
#      a. simplified: p(TAG i = tag i) = arg max p(TAG i = tag i | WORD i = word i)
#          Just find the max probability of p(TAG i = tag i | WORD i = word i)
#      b. HMM VE: p(TAG i = tag i) o= arg max p(WORD i = word i | TAG i = tag i) p(TAG i | TAG i-1)
#          find the max probability of p(WORD i = word i | TAG i = tag i) p(TAG i | TAG i-1)
#          from matrix/list from training result
#      c. HMM MAP: p(tag0, tag1, ..., tag n)
#          o= arg max p(tag0, tag1, ..., tag n) * PRODUCT: p(WORD i = word i | TAG i = tag i)
#          find the max probability series and track back to get tag (with max probability) of each word
#      d. logarithm of the posterior probability:
#          Log(p(TAG i = tag i | WORD i = word i) * P(WORD i = word i) / p(TAG i = tag i))
#          from matrix/list from training result
#      (o= means "proportional to")
#

# (3) a discussion of any problems, assumptions, simplifications, and/or design decisions you made;

#       Laplace Smoothing: use a small const to replace 0 when initiating the matrix/list
#       to avoid "divided by 0" or "log(0)" problems
#
#       In most cases, we don't need to find the exact value of the probability and we just need to find
#        the max value in a list, so we can simplify most calculation by find the proportional factors.
#

# (4) answers to any questions asked below in the assignment.

#       The final test result:
#       command: ./label.py bc.train bc.test
#       result:
#           ==> So far scored 2000 sentences with 29442 words.
#                              Words correct:     Sentences correct:
#              0. Ground truth:      100.00%              100.00%
#                1. Simplified:       93.95%               47.50%
#                    2. HMM VE:       94.45%               50.95%
#                   3. HMM MAP:       94.92%               53.60%
#            ----
#


import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
#
class Solver:
    tag_list = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
    # P(tag)
    tag_frequency_list = [0] * len(tag_list)
    # P(word)
    word_list = []
    # P(word|tag)
    word_tag_matrix = []
    # P(tag|word)
    tag_word_matrix = []
    # P(tag i|tag i-1)
    tag_tag_matrix = []
    sum_word = 0
    # to remove any possibility of divide/0 of log(0)
    laplace_const = 0.0001

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        result = 0
        for i in range(len(sentence)):
            word = sentence[i]
            tag = label[i]
            tag_index = self.tag_list.index(tag)
            if word in self.word_list:
                word_index = self.word_list.index(word)
                # get P(word)
                probability_word = 1.0 * sum(self.word_tag_matrix[word_index]) / self.sum_word
                # get P(tag|word)
                probability_tag_word = 1.0 / sum(self.word_tag_matrix[word_index])
                if self.word_tag_matrix[word_index][tag_index] > 0:
                    probability_tag_word *= self.word_tag_matrix[word_index][tag_index]

            else:
                # get P(word)
                probability_word = 1.0 / self.sum_word
                # get P(tag|word)
                probability_tag_word = 1.0
            # get P(tag)
            probability_tag = self.tag_frequency_list[tag_index]
            # get P(word|tag) = P(tag|word) * P(word) / P(tag)
            probability_word_tag = probability_tag_word * probability_word / probability_tag
            # get log posterior
            if probability_word_tag != 0.0:
                log_posterior = math.log(probability_word_tag, math.e)
            else:
                log_posterior = math.log(0.01, math.e)
            # get sum of log posterior of the sentence
            result += log_posterior
        return result

    # Do the training!
    #
    def train(self, data):
        # initiate the matrix of tag i -> tag i+1 with Laplace smoothing const
        for i in range(0, len(self.tag_list) + 1):
            self.tag_tag_matrix.append([self.laplace_const] * (len(self.tag_list) + 1))

        for line in data:
            for i in range(len(line[0])):
                word = line[0][i]
                tag = line[1][i]

                # build word (frequency) list
                self.sum_word += 1
                if word not in self.word_list:
                    self.word_list.append(word)
                    self.word_tag_matrix.append([self.laplace_const] * len(self.tag_list))
                    self.word_tag_matrix[-1][self.tag_list.index(tag)] += 1
                else:
                    self.word_tag_matrix[self.word_list.index(word)][self.tag_list.index(tag)] += 1

                # build the matrix of tag i -> tag i+1
                # start of the sentence
                if i == 0:
                    # P(start | tag 0)
                    self.tag_tag_matrix[-1][self.tag_list.index(tag)] += 1
                # end of the sentence
                elif i + 1 >= len(line[1]):
                    self.tag_tag_matrix[self.tag_list.index(tag)][-1] += 1
                # other part of the sentence
                else:
                    next_tag = line[1][i + 1]
                    self.tag_tag_matrix[self.tag_list.index(tag)][self.tag_list.index(next_tag)] += 1

                # build tag frequency list
                self.tag_frequency_list[self.tag_list.index(tag)] += 1

        # calculate tag frequency / sum
        frequency_sum = sum(self.tag_frequency_list)
        self.tag_frequency_list = [1.0 * frequency / frequency_sum for frequency in self.tag_frequency_list]

        # calculate tag -> word matrix
        self.update_tag_word_matrix()

    # calculate tag -> word matrix
    def update_tag_word_matrix(self):
        # self.tag_word_matrix.clear()
        self.tag_word_matrix[:] = []
        for i in range(0, len(self.tag_list)):
            temp_sum = sum([row[i] for row in self.word_tag_matrix])
            self.tag_word_matrix.append([1.0 * row[i] / temp_sum for row in self.word_tag_matrix])

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        result = []
        for word in sentence:
            if word in self.word_list:
                index = self.word_list.index(word)
                # get max frequency tag: arg max P(tag|word)
                result.append(self.tag_list[self.word_tag_matrix[index].index(max(self.word_tag_matrix[index]))])
            else:
                # if a word did not appeared before, regard it as "noun" (highest probability)
                result.append("noun")
        return result

    def hmm_ve(self, sentence):
        result = []
        for word in sentence:
            if word in self.word_list:
                index = self.word_list.index(word)
                # simplified based on the P(tag|word)
                # temp_list = list(self.word_tag_matrix[index])
                temp_list = [self.tag_word_matrix[t][index] for t in range(len(self.tag_word_matrix))]
                # get tag i-1
                last_tag = result[-1] if len(result) > 0 else None
                # if start of the sentence, use last line: None -> tag i
                # P(tag|word) * P(tag 0 | start) / P(tag)
                if last_tag is None:
                    for i in range(len(temp_list)):
                        temp_list[i] *= 1.0 * self.tag_tag_matrix[-1][i]
                        # / (sum(self.word_tag_matrix[index])/self.sum_word)  / self.tag_frequency_list[i]
                # other part of the sentence: P(tag|word) * P(tag i | tag i-1) / P(tag)
                else:
                    for i in range(len(temp_list)):
                        temp_list[i] *= \
                            1.0 * self.tag_tag_matrix[self.tag_list.index(last_tag)][i]
                        # / (sum(self.word_tag_matrix[index])/self.sum_word)
                        # * self.tag_frequency_list[self.tag_list.index(last_tag)]
                # get the max one
                result.append(self.tag_list[temp_list.index(max(temp_list))])
            # if a word did not appeared before, regard it as "noun" (highest probability)
            else:
                result.append("noun")
        return result

    # skeleton from viterbi algorithm on wiki:
    # https://en.wikipedia.org/wiki/Viterbi_algorithm
    def hmm_viterbi(self, sentence):
        v = [{}]

        # start of the sentence
        for tag in self.tag_list:
            index = self.tag_list.index(tag)
            if sentence[0] in self.word_list:
                v[0][index] = {
                    # probability = P(tag 0 | start) * P(tag|word)
                    "prob": self.tag_tag_matrix[-1][index] * self.tag_word_matrix[index][
                        self.word_list.index(sentence[0])],
                    "prev": None}
            else:
                # if not in word list, regard it as noun.
                v[0][index] = {
                    "prob": 1.0 if tag == 'noun' else 0,
                    "prev": None}

        # Run Viterbi when t > 0
        for t in range(1, len(sentence)):
            v.append({})
            for tag in self.tag_list:
                index = self.tag_list.index(tag)
                # get max track probability
                max_tr_prob = max(
                    # last probability * P(tag i | tag i-1)
                    v[t - 1][self.tag_list.index(prev_tag)]["prob"]
                    * self.tag_tag_matrix[self.tag_list.index(prev_tag)][index]
                    for prev_tag in self.tag_list
                )

                for prev_tag in self.tag_list:
                    index2 = self.tag_list.index(prev_tag)
                    # find the max track probability
                    if v[t - 1][index2]["prob"] * self.tag_tag_matrix[index2][index] == max_tr_prob:
                        if sentence[t] in self.word_list:
                            # get max probability: P(tag_word) * max track probability
                            max_prob = max_tr_prob * self.tag_word_matrix[index][self.word_list.index(sentence[t])]
                            v[t][index] = {"prob": max_prob, "prev": prev_tag}
                        else:
                            v[t][index] = {
                                "prob": 1 if tag == 'noun' else 0,
                                "prev": prev_tag}
                        break

        opt = []
        # The highest probability
        max_prob = max(value["prob"] for value in v[-1].values())
        previous = None

        # Get most probable state and its backtrack
        for st, data in v[-1].items():
            if data["prob"] == max_prob:
                opt.append(self.tag_list[st])
                previous = self.tag_list[st]
                break

        # Follow the backtrack till the first observation
        for t in range(len(v) - 2, -1, -1):
            opt.insert(0, v[t + 1][self.tag_list.index(previous)]["prev"])
            previous = v[t + 1][self.tag_list.index(previous)]["prev"]
        return opt

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
