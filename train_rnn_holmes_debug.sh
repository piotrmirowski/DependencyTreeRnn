#!/bin/sh

# This is the path that should be edited,
# depending on where the JSON books are stored
PATH_JSON="/Users/piotr/Documents/Projets/Microsoft/Data/GutenbergHolmes/"
PATH_JSON_PIOTR_MAC="/Users/piotr/Documents/Projets/Microsoft/Data/GutenbergHolmes/"

PATH_DATA="./books"
PATH_MODELS="./models"
LIST_VALID="$PATH_DATA/valid.txt"
LIST_TRAIN="$PATH_DATA/train_small.txt"
FILE_MODEL="$PATH_MODELS/GutenbergHolmes_debug.model"

# Train the dependency-parsing model
# I will comment the options later
RnnDependencyTree \
  -rnnlm $FILE_MODEL \
  -train $LIST_TRAIN \
  -valid $LIST_VALID \
  -path-json-books $PATH_JSON \
  -min-word-occurrence 1 \
  -feature-labels-type 1 \
  -hidden 100 \
  -direct 200 \
  -direct-order 3 \
  -bptt 4 \
  -class 100 \
  -debug false
