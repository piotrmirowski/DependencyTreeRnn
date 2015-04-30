#!/bin/sh

# This is the path that should be edited,
# depending on where the JSON books are stored
PATH_SEQUENTIAL=$PWD"/../Data/GutenbergHolmes_Sequential/"

# Get the model filename from the command line
FILE_MODEL=$1

# If we need to debug, change this to "true"
DEBUG_MODE="false"

# Automatic path generation
PATH_DATA="./books"
PATH_MODELS="./models"
FILE_VALID=$PATH_SEQUENTIAL"/Holmes.valid.json.tokens.txt"
FILE_TEST=$PATH_SEQUENTIAL"/Holmes.test.json.tokens.txt"
FILE_SENTENCE_LABELS_VALID=$PATH_DATA"/valid.labels"
FILE_SENTENCE_LABELS_TEST=$PATH_DATA"/test.labels"
echo "RNN model will be stored in $FILE_MODEL..."

# Evaluate the dependency-parsing model on the validation set
RnnDependencyTree \
  -rnnlm $FILE_MODEL \
  -test $FILE_VALID \
  -sentence-labels $FILE_SENTENCE_LABELS_VALID \
  -debug $DEBUG_MODE

# Evaluate the dependency-parsing model on the test set
RnnDependencyTree \
  -rnnlm $FILE_MODEL \
  -test $FILE_TEST \
  -sentence-labels $FILE_SENTENCE_LABELS_TEST \
  -debug $DEBUG_MODE
