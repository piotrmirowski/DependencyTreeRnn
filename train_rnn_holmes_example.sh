#!/bin/sh

# This is the path that should be edited,
# depending on where the JSON books are stored
PATH_JSON="/Users/piotr/Documents/Projets/Microsoft/Data/GutenbergHolmes/"
PATH_JSON_PIOTR_MAC="/Users/piotr/Documents/Projets/Microsoft/Data/GutenbergHolmes/"

DEP_LABELS=2
MIN_WORD_OCCURRENCE=5
RNN_HIDDENS=100
RNN_CLASSES=500
NGRAM_SIZE_MB=2000
NGRAM_ORDER=3
BPTT_ORDER=4


# Automatic path generation
PATH_DATA="./books"
PATH_MODELS="./models"
LIST_VALID=$PATH_DATA"/valid.txt"
LIST_TRAIN=$PATH_DATA"/train.txt"
FILE_SENTENCE_LABELS=$PATH_DATA"/valid.labels"
FILE_MODEL=$PATH_MODELS"/GutenbergHolmes_p"$DEP_LABELS
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
  -train $LIST_TRAIN \
  -valid $LIST_VALID \
  -sentence-labels $FILE_SENTENCE_LABELS \
  -path-json-books $PATH_JSON \
  -min-word-occurrence $MIN_WORD_OCCURRENCE \
  -feature-labels-type $DEP_LABELS \
  -hidden $RNN_HIDDENS \
  -direct $NGRAM_SIZE_MB \
  -direct-order $NGRAM_ORDER \
  -bptt $BPTT_ORDER \
  -class $RNN_CLASSES \
  -debug false
