#!/bin/sh

# This is the path that should be edited,
# depending on where the JSON books are stored
PATH_JSON="/Users/piotr/Documents/Projets/Microsoft/Data/GutenbergHolmes/"

# Get the model file name from the argument and the vocabulary file
FILE_MODEL=$1
FILE_VOCAB=$2
DEP_LABELS=$3
FEATURE_GAMMA=$4

# If we need to debug, change this to "true"
DEBUG_MODE="false"

# Automatic path generation
PATH_DATA="./books"
PATH_MODELS="./models"
LIST_VALID=$PATH_DATA"/valid.txt"
LIST_TEST=$PATH_DATA"/test.txt"
FILE_SENTENCE_LABELS_VALID=$PATH_DATA"/valid.labels"
FILE_SENTENCE_LABELS_TEST=$PATH_DATA"/test.labels"
echo "RNN model is read from $FILE_MODEL..."

# Test the dependency-parsing model on the validation data
RnnDependencyTree \
  -rnnlm $FILE_MODEL \
  -test $LIST_VALID \
  -sentence-labels $FILE_SENTENCE_LABELS_VALID \
  -path-json-books $PATH_JSON \
  -vocab $FILE_VOCAB \
  -debug $DEBUG_MODE \
  -feature-labels-type $DEP_LABELS \
  -feature-gamma $FEATURE_GAMMA

# Test the dependency-parsing model on the test data
RnnDependencyTree \
  -rnnlm $FILE_MODEL \
  -test $LIST_TEST \
  -sentence-labels $FILE_SENTENCE_LABELS_TEST \
  -path-json-books $PATH_JSON \
  -vocab $FILE_VOCAB \
  -debug $DEBUG_MODE \
  -feature-labels-type $DEP_LABELS \
  -feature-gamma $FEATURE_GAMMA
