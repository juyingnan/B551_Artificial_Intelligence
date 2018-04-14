#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors:
# Name: Yingnan Ju; ID: yiju
# Name: Yue Chen; ID: yc59
# (based on skeleton code by D. Crandall, Oct 2017)
####
# Put your report here!!
#
# Report by Yingnan Ju (yiju) and Yue Chen (yc59)
#
# I wrote a test function to test all 20 png files to get the performance of 3 different methods.
# Final test result of all 20 png files:
#                   Letter correct:     Sentences correct:
#     1. Simplified:        96.93%                  30%
#         2. HMM VE:        99.11%                  80%
#        3. HMM MAP:        99.01%                  85%
# ----

# (1) a description of how you formulated the problem,
# including precisely defining the abstractions (e.g. HMM formulation); #

#      a. simplified: p(LETTER i = letter i) = arg max p(LETTER i = letter i | PIXELS = pixels)
#           Just find the max probability of p(LETTER i = letter i | PIXELS = pixels)
#
#      b. HMM VE: p(LETTER i = letter i) = arg max p(LETTER i = letter i | PIXEL = pixel)
#          = arg max p(LETTER i = letter i, PIXEL = pixel) / p(PIXEL = pixel)
#          = arg max p(PIXEL = pixel | LETTER i = letter i) p(LETTER i = letter i) / p(PIXEL = pixel)
#          = arg max p(PIXEL = pixel | LETTER i = letter i) p(LETTER i | LETTER i-1) p(LETTER i-1 = letter i-1)
#                   / p(PIXEL = pixel)
#          o= arg max p(PIXEL = pixel | LETTER i = letter i) p(LETTER i | LETTER i-1)
#          because we don't need to calculate the exact probability but only need to find the highest probability
#           (o= means "proportional to")
#
#      c. HMM MAP: p(letter0, letter1, ..., letter n)
#           = arg max p(letter0, letter1, ..., letter n | pixel0, pixel1, ..., pixel n)
#           = arg max p(pixel0, pixel1, ..., pixel n | letter0, letter1, ..., letter n)
#               * p(letter0, letter1, ..., letter n) / p(pixel0, pixel1, ..., pixel n)
#           o= arg max p(pixel0, pixel1, ..., pixel n | letter0, letter1, ..., letter n)
#               * p(letter0, letter1, ..., letter n)
#           = arg max p(letter0, letter1, ..., letter n) * PRODUCT: p(PIXEL i = pixel i | LETTER i = letter i)
#           (o= means "proportional to")
#

# (2) a brief description of how your program works;

#      in training part, get:
#          p(LETTER i = letter i | PIXEL = pixel) from a matrix
#           (Y/N means '*' or ' ', Y = 1 + laplace const; N = laplace const):
#
#                   |   letter 0   letter 1    ...     letter n
#           --------|--------------------------------------------
#           point 0 |   Y/N         Y/N        ...
#           point 1 |
#            ...    |
#           point n |                          ...      Y/N
#           sum     |   sum 0       sum 1      ...      sum n
#
#           by calculating:
#               if point n from pixel|letter m == Y:
#                   likelihood *= point n / sum m
#               if point n from pixel|letter m == N:
#                   likelihood *= (1 - point n / sum m)
#
#          p(PIXEL = pixel | LETTER i = letter i)
#           difficult to calculate, replace it with corresponding value from p(LETTER i = letter i | PIXEL = pixel)
#
#          p(LETTER i | LETTER i-1)  ,a matrix:
#
#                   |   letter 0   letter 1   ...     letter n       end (this part has not been used yet)
#           --------|-----------------------------------------------------
#         letter 0  |   p(l0|l0)   p(l1|l0) ...                     p(end|l0)
#         letter 1  |
#            ...    |                                               ...
#         letter n  |                       ...       p(ln|ln)      p(end|ln)
#            start  |   p(l0|start)         ...       p(ln|start)   p(end|start)
#
#          p(LETTER i = letter i), a frequency list:
#           [p(letter 0), p(letter 1), ... , p(letter n)]
#
#      a. simplified: p(LETTER i = letter i) = arg max p(LETTER i = letter i | PIXELS = pixels)
#          Just find the max probability by calculating:
#               if point n from pixel|letter m == Y:
#                   likelihood *= point n / sum m
#               if point n from pixel|letter m == N:
#                   likelihood *= (1 - point n / sum m)
#      b. HMM VE: p(LETTER i = letter i) o= arg max p(PIXEL = pixel | LETTER i = letter i) p(LETTER i | LETTER i-1)
#          find the max probability of arg max p(PIXEL = pixel | LETTER i = letter i) p(LETTER i | LETTER i-1)
#          from matrix/list from training result
#      c. HMM MAP: p(letter0, letter1, ..., letter n)
#          o= arg max p(letter0, letter1, ..., letter n) * PRODUCT: p(PIXEL i = pixel i | LETTER i = letter i)
#          find the max probability series and track back to get letter (with max probability) of each set of pixels
#      (o= means "proportional to")
#

# (3) a discussion of any problems, assumptions, simplifications, and/or design decisions you made;

#       Laplace Smoothing: use a small const to replace 0 when initiating the matrix/list
#       to avoid "divided by 0" or "log(0)" problems
#
#       In most cases, we don't need to find the exact value of the probability and we just need to find
#        the max value in a list, so we can simplify most calculation by find the proportional factors.
#
#       One methods to improve accuracy:
#           (a) the way to calculate: p(LETTER i = letter i | PIXEL = pixel) has a high priority for those '*' points
#               and a low priority for those ' ' points. The difference between low and high priority affects the
#               result and if I increase low priority by (* count of points / n = 9), the result is the best:
#               space_factor = CHARACTER_WIDTH * CHARACTER_HEIGHT / 9.0
#               if point n from pixel|letter m == Y:
#                   likelihood *= point n / sum m
#               if point n from pixel|letter m == N:
#                   likelihood *= (1 - space_factor * point n / sum m)
#               This method will improve the result of all three methods.
#
#           (b) p(PIXEL = pixel | LETTER i = letter i) is difficult to calculate, so we replace it with corresponding
#               value from p(LETTER i = letter i | PIXEL = pixel) and that also got a good result.
#

# (4) answers to any questions asked below in the assignment.

#       The final test result:
#       command: /ocr.py courier-train.png test-strings.txt test-0-0.png
#       result:
#               Final test result of all 20 png files:
#                                     Letter correct:     Sentences correct:
#                1. Simplified:        96.93%                  30%
#                    2. HMM VE:        99.11%                  80%
#                   3. HMM MAP:        99.01%                  85%
#            ----
#       Uncomment test() function at the end of this file to get the result.
#       The print() function is for Python 3 and Python 2 will regard the result as tuples
#       so the printed lines will be with brackets.


from PIL import Image, ImageDraw, ImageFont
import sys
import math

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25

# for ocr train
character_list = []
pixel_character_matrix = []
character_pixel_matrix = []
# Laplace smoothing const, to avoid possibility that divide/0 or log(0)
# 0.1 is the best (magic) value: tried from 0.01->0.2 and 0.2-> 1.0
LAPLACE_const = 0.1
# space factor is to improve probability of "SPACE". Accuracy increased a lot!
# n=9: best for HMM VE and HMM MAP; 8 is best for NAIVE
space_factor = CHARACTER_WIDTH * CHARACTER_HEIGHT / 9.0

# for bc train
character_frequency_list = []
character_character_matrix = []


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    # print(im.size)
    # print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [["".join(['*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg + CHARACTER_WIDTH)]) for y in
                    range(0, CHARACTER_HEIGHT)], ]
    return result


def load_training_letters(fname):
    train_letters_library = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return {train_letters_library[i]: letter_images[i] for i in range(0, len(train_letters_library))}


# Read in training or test data file
# modified from part1 - label.py; author: D. Crandall
# Divide the sentence into characters
def read_data(fname):
    exemplars = []
    result = []
    file = open(fname, 'r')
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += [(data[0::2]), ]
    for line in exemplars:
        new_line = []
        for word in line:
            for char in word:
                new_line.append(char)
            new_line.append(' ')
        new_line.pop()
        result.append(new_line)

    return result


# train from corpus file
def train_bc(data):
    # get character frequency list
    # character_frequency_list.clear()
    character_frequency_list[:] = []
    for i in range(len(character_list)):
        character_frequency_list.append(1)

    # initiate matrix for P(character i | character i-1)
    # 1 : 97.32 v
    # 0.1: 96.8
    # character_character_matrix.clear()
    character_character_matrix[:] = []
    for i in range(0, len(character_list) + 1):
        character_character_matrix.append([1] * (len(character_list) + 1))

    for line in data:
        for i in range(len(line)):
            character = line[i]
            if character in character_list:
                # build matrix for P(character i | character i-1)
                # start of the sentence
                # P(character 0| start)
                if i == 0:
                    character_character_matrix[-1][character_list.index(character)] += 1
                # end of the sentence
                elif i + 1 >= len(line):
                    character_character_matrix[character_list.index(character)][-1] += 1
                # other part
                # P(character i | character i-1)
                else:
                    # next_character = ' ' if line[i + 1] not in character_list else line[i + 1]
                    next_character = line[i + 1]
                    # if character not in character list, ignore
                    if next_character in character_list:
                        character_character_matrix[character_list.index(character)][
                            character_list.index(next_character)] += 1

                # build character frequency list
                character_frequency_list[character_list.index(character)] += 1

    # calculate tag frequency
    frequency_sum = sum(character_frequency_list)
    for i in range(len(character_frequency_list)):
        character_frequency_list[i] /= 1.0 * frequency_sum


# get a list of a matrix
def get_flat_list(matrix):
    temp_flat_list = []
    for row in matrix:
        temp_flat_list.extend(row)
    return temp_flat_list


# train ocr letters from lixels of letters
def train_ocr(letters):
    print("OCR TRAINING...")
    pixel_count = CHARACTER_WIDTH * CHARACTER_HEIGHT
    character_count = len(letters)

    # clear P(character|pixels) and character list
    # pixel_character_matrix.clear()
    pixel_character_matrix[:] = []
    # character_list.clear()
    character_list[:] = []

    # initiate P(character|pixels) and character list
    for i in range(pixel_count):
        pixel_character_matrix.append([LAPLACE_const] * character_count)
    for key in letters.keys():
        character_list.append(key)

    # build P(character|pixels) matrix
    for key in letters.keys():
        temp_flat_list = get_flat_list(letters[key])
        for i in range(len(temp_flat_list)):
            if temp_flat_list[i] != ' ':
                pixel_character_matrix[i][character_list.index(key)] += 1
    # get sum of the matrix and append it at the last line
    pixel_character_matrix.append(
        [sum([pixel_character_matrix[i][j] for i in range(pixel_count)]) for j in range(character_count)])

    # calculate character -> pixel matrix: P(pixels|character)
    update_character_pixel_matrix()


# calculate character -> pixel matrix: P(pixels|character)
def update_character_pixel_matrix():
    # character_pixel_matrix.clear()
    character_pixel_matrix[:] = []
    for i in range(0, len(character_list)):
        temp_sum = sum([row[i] for row in pixel_character_matrix])
        character_pixel_matrix.append([1.0 * row[i] / temp_sum for row in pixel_character_matrix])


# naive bayes
def simplified(letters_in_pixel):
    result = ''
    for letter in letters_in_pixel:
        # get the probability that pixels could be which character
        likelihood_list = get_likelihood_pixel_to_letter(letter)
        result += character_list[likelihood_list.index(max(likelihood_list))]
    # print(result)
    return result


# get the probability that pixels could be which character
# inspired by: http://www.codingvision.net/ai/c-naive-bayes-basic-ocr-w-example
# improved it a lot
def get_likelihood_pixel_to_letter(letter):
    likelihood_list = []
    for i in range(len(character_list)):
        # initiate likelihood
        likelihood_list.append(1.0 / len(character_list))
        flat_list = get_flat_list(letter)
        for j in range(len(flat_list)):
            # for each pixel, get the probability
            # space priority is lower, but not too low
            # that is why space_factor is needed: to improve/balance the priority
            if flat_list[j] == ' ':
                likelihood_list[i] *= (1 - space_factor * pixel_character_matrix[j][i] / pixel_character_matrix[-1][i])
            # '*' priority is higher
            else:
                likelihood_list[i] *= 1.0 * pixel_character_matrix[j][i] / pixel_character_matrix[-1][i]
    total_likelihood = sum(likelihood_list)
    for i in range(len(likelihood_list)):
        likelihood_list[i] /= 1.0 * total_likelihood
    return likelihood_list


# get the probability that character could be specific combination of pixels
# currently not used#
# replaced by get_likelihood_pixel_to_letter(letter)
def get_likelihood_letter_to_pixel(letters, pixels):
    likelihood_list = []
    pixels_list = [1.1 if pixel != ' ' else 0.1 for pixel in get_flat_list(pixels)]
    temp_sum = sum(pixels_list)
    # pixels_list = get_flat_list(pixels)
    for letter in letters:
        index = letters.index(letter)
        flat_letter_pixels = get_flat_list(train_letters[letter])
        likelihood_list.append(0.0)
        for j in range(0, len(flat_letter_pixels)):
            if flat_letter_pixels[j] != ' ':
                likelihood_list[index] -= math.log(1 - pixels_list[j] / temp_sum, math.e)
            else:
                likelihood_list[index] -= math.log(pixels_list[j] / temp_sum, math.e)
                # if flat_letter_pixels[j] != pixels_list[j]:
                #     likelihood_list[index] *= 0.55
    result = [math.exp(likelihood) for likelihood in likelihood_list]
    total_likelihood = sum(result)
    result = [likelihood / total_likelihood for likelihood in result]
    return result


# naive bayes + probability from corpus file
def hmm_ve(letters_in_pixel):
    character_count = len(character_list)
    result = ''
    for letter in letters_in_pixel:
        likelihood_list = []
        # get character i-1
        last_character = result[-1] if len(result) > 0 else None
        for i in range(character_count):
            likelihood_list.append(1.0 / character_count)
            # if start of the sentence:
            if last_character is None:
                # P(character 0| start)
                likelihood_list[i] = character_character_matrix[-1][i]
            else:
                # P(character i| character i-1)
                likelihood_list[i] = \
                    character_character_matrix[character_list.index(last_character)][i]

            # naive part
            # same with the one before, but based on the HMM probability
            flat_list = get_flat_list(letter)
            for j in range(len(flat_list)):
                if flat_list[j] == ' ':
                    likelihood_list[i] *= \
                        (1 - space_factor * pixel_character_matrix[j][i] / pixel_character_matrix[-1][i])
                else:
                    likelihood_list[i] *= 1.0 * pixel_character_matrix[j][i] / pixel_character_matrix[-1][i]
        total_likelihood = sum(likelihood_list)
        for i in range(len(likelihood_list)):
            likelihood_list[i] /= 1.0 * total_likelihood
        result += character_list[likelihood_list.index(max(likelihood_list))]
    # print(result)
    return result


def hmm_map(letters_in_pixel):
    v = [{}]
    # use pixels->character to replace character->pixels
    # likelihood_list = get_likelihood_letter_to_pixel(character_list, test_letters[0])

    # start of the sentence
    likelihood_list = get_likelihood_pixel_to_letter(letters_in_pixel[0])
    for character in character_list:
        index = character_list.index(character)
        v[0][index] = {
            # P(character 0|start) * P(character|pixels)
            "prob": character_character_matrix[-1][index] * likelihood_list[index],
            "prev": None}

    # Run Viterbi when t > 0
    for t in range(1, len(letters_in_pixel)):
        v.append({})
        # likelihood_list = get_likelihood_letter_to_pixel(character_list, test_letters[t])
        # get P(character|pixels)
        likelihood_list = get_likelihood_pixel_to_letter(letters_in_pixel[t])
        for character in character_list:
            index = character_list.index(character)
            # get max track probability
            max_tr_prob = max(
                v[t - 1][character_list.index(prev_character)]["prob"]
                * character_character_matrix[character_list.index(prev_character)][index]
                for prev_character in character_list
            )

            for prev_character in character_list:
                index2 = character_list.index(prev_character)
                # get max probability
                if v[t - 1][index2]["prob"] * character_character_matrix[index2][index] == max_tr_prob:
                    max_prob = max_tr_prob * likelihood_list[index]
                    v[t][index] = {"prob": max_prob, "prev": prev_character}
                    break

    opt = []
    # The highest probability
    max_prob = max(value["prob"] for value in v[-1].values())
    previous = None

    # Get most probable state and its backtrack
    for st, data in v[-1].items():
        if data["prob"] == max_prob:
            opt.append(character_list[st])
            previous = character_list[st]
            break

    # Follow the backtrack till the first observation
    for t in range(len(v) - 2, -1, -1):
        opt.insert(0, v[t + 1][character_list.index(previous)]["prev"])
        previous = v[t + 1][character_list.index(previous)]["prev"]
    result = ''
    for letter in opt:
        result += letter
    # print(result)
    return result


#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

# Below is just some sample code to show you how the functions above work.
# You can delete them and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
# print("\n".join([r for r in train_letters['a']]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
# print("\n".join([r for r in test_letters[2]]))

# train ocr letters pixels
train_ocr(train_letters)

# train corpus file
bc_data = read_data("bc.train.tiny")
train_bc(bc_data)

simple_result = simplified(test_letters)
hmm_ve_result = hmm_ve(test_letters)
hmm_map_result = hmm_map(test_letters)
print(" Simple: " + simple_result)
print(" HMM VE: " + hmm_ve_result)
print("HMM MAP: " + hmm_map_result)


def test_all():
    global test_letters
    print("TESTING...")
    test_file = open('test-strings.txt', 'r')
    test_results = []
    temp_result = [[], [], []]
    print("LOADING TEST RESULT FILE...")
    for test_result in test_file:
        test_results.append(test_result)
    print("OCR TEST PICTURE FILES...")
    for file_num in range(20):
        file_name = 'test-' + file_num.__str__() + '-0.png'
        test_letters = load_letters(file_name)
        temp_result[0].append(simplified(test_letters))
        temp_result[1].append(hmm_ve(test_letters))
        temp_result[2].append(hmm_map(test_letters))
    print("STATISTICS...")
    for m in range(3):
        total_letter_count = 0
        total_letter_correct = 0
        sentence_count = 0
        sentence_correct = 0
        for n in range(20):
            sentence_count += 1
            letter_count = 0
            letter_correct = 0
            for l in range(min(len(test_results[n]), len(temp_result[m][n]))):
                letter_count += 1
                if test_results[n][l] == temp_result[m][n][l]:
                    letter_correct += 1
            if letter_correct == letter_count:
                sentence_correct += 1
            total_letter_correct += letter_correct
            total_letter_count += letter_count
            # print('sentence ', n, ': ', letter_count, 'letters , ', format(1.0*letter_correct / letter_count, '0.2%'))
        print('total_letter_count ', total_letter_count, ' ',
              format(1.0 * total_letter_correct / total_letter_count, '0.2%'))
        print('sentence_count ', sentence_count, ' ', format(1.0 * sentence_correct / sentence_count, '0.2%'))

# test codes
# comment after testing
# test_all()
