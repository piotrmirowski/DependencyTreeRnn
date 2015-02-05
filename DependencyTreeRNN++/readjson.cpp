// Copyright (c) 2014 Piotr Mirowski. All rights reserved.
//                    piotr.mirowski@computer.org

#include <stdio.h>
#include <iostream>
#include <string>
#include "json.h"
#include "readjson.h"
#include "CorpusUnrollsReader.h"

using namespace std;

/**
 * Constructor: read a text file in JSON format.
 * If required, insert words and labels to the vocabulary.
 * If required, insert tokens into the current book.
 */
ReadJson::ReadJson(const string &filename,
                   CorpusUnrolls &corpus,
                   bool insert_vocab,
                   bool read_book,
                   bool merge_label_with_word) {
  
  // Used for parsing the JSON data file
  char *errorPos = 0;
  const char *errorDesc = 0;
  int errorLine = 0;
  block_allocator allocator(10000000); // 1MB blocks
  
  // Read the file with JSON data
  FILE *fin = fopen(filename.c_str(), "rb");
  if (fin == NULL) {
    cerr << "Could not open file " << filename << "...\n";
  }
  char *source = (char *)allocator.malloc(100000000); // 100MB
  int a = 0;
  while (!feof(fin)) {
    char ch = fgetc(fin);
    source[a] = ch;
    a++;
  }
  fclose(fin);
  
  // Parse the JSON
  _root = json_parse(source,
                     &errorPos, &errorDesc, &errorLine,
                     &allocator);
  if (_root == NULL) {
    cerr << "\nError at line " << errorLine << endl;
    cerr << errorDesc << endl << errorPos << endl;
  }
  
  // Pointer to the current book
  BookUnrolls *book = &(corpus.m_currentBook);
  
  // First, iterate over sentences
  int numSentences = 0;
  for (json_value *s = _root->first_child;
       s;
       s = s->next_sibling) {
    int numUnrollsInThatSentence = 0;
    bool isNewSentence = true;
    // Second, iterate over unrolls in each sentence
    for (json_value *u = s->first_child;
         u;
         u = u->next_sibling) {
      bool isNewUnroll = true;
      // Third, iterate over tokens in each unroll
      for (_token = u->first_child;
           _token;
           _token = _token->next_sibling) {
        // Process the token to get:
        // its position in sentence,
        // word, discount and label
        ProcessToken();
        string tokenWord(_tokenWord);
        if (merge_label_with_word) {
          tokenWord = string(_tokenWord) + ":" + string(_tokenLabel);
        }
        // Shall we insert new words/labels
        // into the vocabulary?
        if (insert_vocab) {
          if (merge_label_with_word) {
            // Insert concatenated word and label to vocabulary
            corpus.InsertWord(tokenWord, _tokenDiscount);
          } else {
            // Insert word and label to two different vocabularies
            corpus.InsertWord(tokenWord, _tokenDiscount);
            corpus.InsertLabel(_tokenLabel);
          }
        }
        // Insert new words to the book
        int wordIndex = 0, labelIndex = 0;
        if (merge_label_with_word) {
          wordIndex = corpus.LookUpWord(tokenWord);
          labelIndex = corpus.LookUpLabel(_tokenLabel);
        } else {
          wordIndex = corpus.LookUpWord(tokenWord);
        }
        book->AddToken(isNewSentence, isNewUnroll,
                       _tokenPos, wordIndex,
                       _tokenDiscount, labelIndex);
        // We are no longer at beginning of a sentence or unroll
        isNewSentence = false;
        isNewUnroll = false;
      }
      numUnrollsInThatSentence++;
    }
    _numUnrollsPerSentence.push_back(numUnrollsInThatSentence);
    numSentences++;
  }
  cout << "\nReadJSON: " << filename << endl;
  cout << "          (" << NumSentences() << " sentences, including empty ones; ";
  cout << book->NumTokens() << " tokens)\n";
  if (insert_vocab) {
    cout << "          Corpus now contains " << corpus.NumWords()
    << " words and " << corpus.NumLabels() << " labels\n";
  }
}


/**
 * Go to a specific sentence
 */
bool ReadJson::GoToSentence(int n) {
  // Safety check
  if (n >= NumSentences())
    return false;
  // Do nothing?
  if (_sentenceIndex == n) {
    ResetUnroll();
    return true;
  }
  // Do we need to reset sentence position?
  if (_sentenceIndex > n) {
    _sentenceIndex = NumSentences();
    NextSentence();
  }
  // Iterate through the list of sentences
  while (_sentenceIndex < n)
    NextSentence();
  return true;
}


/**
 * Go to the next sentence
 */
int ReadJson::NextSentence() {
  int n = NumSentences();
  if (_sentenceIndex >= n - 1) {
    // Return to sentence 0...
    ResetSentence();
  } else {
    // ... or simply go to next sentence?
    _sentence = _sentence->next_sibling;
    _sentenceIndex++;
    // Reset the unroll
    ResetUnroll();
  }
  return _sentenceIndex;
}


/**
 * Go to the next unroll in the current sentence
 */
int ReadJson::NextUnrollInSentence() {
  int n = NumUnrolls(_sentenceIndex);
  if (_unrollIndex >= n - 1) {
    // Return to unroll 0 in the current sentence...
    ResetUnroll();
  } else {
    // ... or simply go to next unroll?
    _unroll = _unroll->next_sibling;
    _unrollIndex++;
    // Reset the token
    ResetToken();
  }
  return _unrollIndex;
}


/**
 * Go to the next unroll in the current sentence.
 * Here, we do not loop over but stop (return -1)
 * when the end of the unroll is reached.
 */
int ReadJson::NextTokenInUnroll() {
  // If we have reach the end of sentence
  if (_tokenIndex < 0)
    return -1;
  // Then, go to the next token
  _token = _token->next_sibling;
  _tokenIndex++;
  // Process current token
  ProcessToken();
  if (_token == NULL) {
    return -1;
  } else {
    return _tokenIndex;
  }
}


/**
 * Reset the unroll in the current sentence
 */
void ReadJson::ResetSentence() {
  _sentence = _root->first_child;
  _sentenceIndex = 0;
  ResetUnroll();
}


/**
 * Reset the unroll in the current sentence
 */
void ReadJson::ResetUnroll() {
  _unroll = _sentence->first_child;
  _unrollIndex = 0;
  ResetToken();
}


/**
 * Reset the token in the current sentence and unroll
 */
void ReadJson::ResetToken() {
  _token = _unroll->first_child;
  _tokenIndex = 0;
  ProcessToken();
}


/**
 * Parse a token in the JSON parse tree to fill token data
 */
void ReadJson::ProcessToken() {
  // Safety check
  if (_token == NULL) {
    _tokenPos = -1;
    _tokenWord = NULL;
    _tokenDiscount = 0;
    _tokenLabel = NULL;
    return;
  }
  // Get the position of the token in the unroll
  json_value *element = _token->first_child;
  _tokenPos = element->int_value;
  // Get the string of the word in the token in the unroll
  element = element->next_sibling;
  _tokenWord = element->string_value;
  // Get the discount of the token in the unroll
  element = element->next_sibling;
  _tokenDiscount = (double) 1.0 / (element->int_value);
  // Get the label of the token in the unroll
  element = element->next_sibling;
  _tokenLabel = element->string_value;
}


/**
 * Parse a corpus in the JSON parse tree and print it
 */
void ReadJson::TraverseCorpus(bool verbose) {
  // Safety check
  if (_root->type != JSON_ARRAY)
    cerr << "Error: corpus is not a JSON array: "
    << _root->type << endl;
  // Loop over sentences
  ResetSentence();
  int numSentences = NumSentences();
  for (int i = 0; i < numSentences; i++) {
    if (verbose)
      printf("### Sentence %d:\n", _sentenceIndex);
    // Safety check
    if (_sentence->type != JSON_ARRAY)
      cerr << "Error: sentence is not a JSON array: "
      << _sentence->type << endl;
    // Loop over unrolls
    ResetUnroll();
    int n_unrolls = NumUnrolls(i);
    for (int j = 0; j < n_unrolls; j++) {
      // Safety check
      if (_unroll->type != JSON_ARRAY)
        cerr << "Error: unroll is not a JSON array: "
        << _unroll->type << endl;
      if (verbose)
        printf("#   Unroll:");
      // Loop over the tokens
      bool ok = true;
      while (ok) {
        // Safety checks
        if (_token->type != JSON_ARRAY)
          cerr << "Error: token is not a JSON array: "
          << _token->type << endl;
        CheckToken();
        if (verbose)
          printf(" %d:%s:%s(%.3f)",
                 CurrentTokenNumberInSentence(),
                 CurrentTokenWord(),
                 CurrentTokenLabel(),
                 CurrentTokenDiscount());
        ok = (NextTokenInUnroll() >= 0);
      }
      if (verbose)
        printf("\n");
      NextUnrollInSentence();
    }
    NextSentence();
  }
}


/**
 * Parse a token in the JSON parse tree to check its structure
 */
void ReadJson::CheckToken() {
  // Safety check
  if (_token == NULL) {
    _tokenPos = -1;
    _tokenWord = NULL;
    _tokenDiscount = 0;
    _tokenLabel = NULL;
    return;
  }
  json_value *element = _token->first_child;
  if (element->type != JSON_INT)
    cerr << "Error: not an integer token position at pos 0: "
    << element->type << endl;
  element = element->next_sibling;
  if (element->type != JSON_STRING)
    cerr << "Error: not a word at pos 1: "
    << element->type << endl;
  element = element->next_sibling;
  if (element->type != JSON_INT)
    cerr << "Error: not a discount at pos 2: "
    << element->type << endl;
  element = element->next_sibling;
  if (element->type != JSON_STRING)
    cerr << "Error: not a label at pos 3: "
    << element->type << endl;
}
