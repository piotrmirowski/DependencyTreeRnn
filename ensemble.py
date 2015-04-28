# first arg gold, following ones files with scores to ensemble

import sys

goldFile = sys.argv[1]

answers = []

for line in open(goldFile).readlines():
    answers.append(int(line.strip()))

print "loaded " + str(len(answers)) + " answers"

# an array with an array per model to be ensebled
individualSentencePredictions = []

for file in sys.argv[2:]:
    sentencePredictions = []
    for line in open(file).readlines():
        sentencePredictions.append(float(line.strip()))
    
    individualSentencePredictions.append(sentencePredictions)

# now for each answer
# take the scores for 5 sentence predictions
# add them
# pick the highest one and compare
correct = 0.0
indiCounter= 0
for answer in answers:
    maxScore = float("-inf")
    bestAnswer = None
    for i in xrange(5):
        scoreSum = 0.0
        for preds in individualSentencePredictions:
            scoreSum += preds[indiCounter]
        #print scoreSum
        if scoreSum > maxScore:
            maxScore = scoreSum
            bestAnswer = i

        indiCounter += 1
    #print bestAnswer
    #print maxScore
    if answer == bestAnswer:
        correct += 1

print "accuracy: " + str(correct/len(answers))
    
