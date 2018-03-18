# FIND-SAlgorithm.

# 1. Initialize h to the most specific hypothesis in H
# 2. Foreachpositivetraininginstancex
# For each attribute constraint a, in h
# If the constraint a, is satisfied by x
# Then do nothing
# Else replace a, in h by the next more general constraint that is satisfied by x
# 3. Outputhypothesish

from constants import NO_VALUE, ANY_VALUE


def find_s(training_examples):
    h = [NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE]
    for example in training_examples:
        if example[1] == 1:
            tupla = example[0]
            for i in range(0, 5):
                if h[i] == NO_VALUE:
                    h[i] = tupla[i]
                elif h[i] != tupla[i]:
                    h[i] = ANY_VALUE

    return h


def find_s_target(training_examples, target):
    total_evaluados = 0
    hay_match = False
    positivos_evaluados = 0
    h = [NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE]
    for index, example in enumerate(training_examples):
        if example[1] == 1:
            positivos_evaluados += 1
            tupla = example[0]
            for i in range(0, 5):
                if h[i] == NO_VALUE:
                    h[i] = tupla[i]
                elif h[i] != tupla[i]:
                    h[i] = ANY_VALUE

            if match_target(h, target):
                hay_match = True
                total_evaluados = index + 1
                break

    return h, hay_match, total_evaluados, positivos_evaluados


def match_target(h, target):
    match = True
    largo_h = len(h)
    index = 0

    while index < largo_h and match:
        match = h[index] == target[index]
        index += 1

    return match
