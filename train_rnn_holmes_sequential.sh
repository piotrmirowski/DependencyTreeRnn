#!/bin/sh

# This is the path that should be edited,
# depending on where the JSON books are stored
PATH_SEQUENTIAL=$PWD"/../Data/GutenbergHolmes_Sequential/"

# Define the minimum number of word occurrences as 5 and use existing vocabulary file
MIN_WORD_OCCURRENCE=5
RNN_HIDDENS=200
RNN_CLASSES=250
NGRAM_SIZE_MB=2000
NGRAM_ORDER=3
BPTT_ORDER=5

# If we need to debug, change this to "true"
DEBUG_MODE="false"

# Automatic path generation
PATH_DATA="./books"
PATH_MODELS="./models"
FILE_TRAIN=$PATH_SEQUENTIAL"/Holmes.train.json.tokens.txt"
FILE_VALID=$PATH_SEQUENTIAL"/Holmes.valid.json.tokens.txt"
FILE_SENTENCE_LABELS=$PATH_DATA"/valid.labels"
FILE_VOCAB=$PATH_DATA"/vocab_mw"$MIN_WORD_OCCURRENCE"_sequential.txt"
FILE_MODEL=$PATH_MODELS"/GutenbergHolmes_seq"
FILE_MODEL=$FILE_MODEL"_mw"$MIN_WORD_OCCURRENCE
FILE_MODEL=$FILE_MODEL"_h"$RNN_HIDDENS
FILE_MODEL=$FILE_MODEL"_c"$RNN_CLASSES
FILE_MODEL=$FILE_MODEL"_m"$NGRAM_SIZE_MB
FILE_MODEL=$FILE_MODEL"_d"$NGRAM_ORDER
FILE_MODEL=$FILE_MODEL"_b"$BPTT_ORDER
FILE_MODEL=$FILE_MODEL".model"
echo "RNN model will be stored in $FILE_MODEL..."

# Train the dependency-parsing model
RnnDependencyTree \
  -rnnlm $FILE_MODEL \
  -train $FILE_TRAIN \
  -valid $FILE_VALID \
  -sentence-labels $FILE_SENTENCE_LABELS \
  -min-word-occurrence $MIN_WORD_OCCURRENCE \
  -hidden $RNN_HIDDENS \
  -direct $NGRAM_SIZE_MB \
  -direct-order $NGRAM_ORDER \
  -bptt $BPTT_ORDER \
  -bptt-block 1 \
  -class $RNN_CLASSES \
  -debug $DEBUG_MODE
#  -vocab $FILE_VOCAB

