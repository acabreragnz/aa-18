#FIND-SAlgorithm.

#1. Initialize h to the most specific hypothesis in H 
#2. Foreachpositivetraininginstancex
	# For each attribute constraint a, in h
		#If the constraint a, is satisfied by x
			#Then do nothing
		#Else replace a, in h by the next more general constraint that is satisfied by x
#3. Outputhypothesish

from constants import NO_VALUE, ANY_VALUE


def find_s(training_examples):
	h = [NO_VALUE,  NO_VALUE,  NO_VALUE, NO_VALUE, NO_VALUE]
	for example in training_examples:
		if example[1] == 1:
			tupla = example[0]
			for i in range(0,5):
				if h[i] == NO_VALUE:
					h[i] = tupla[i]
				elif h[i] != tupla[i]:
					h[i] = ANY_VALUE
				
				
	return h				






