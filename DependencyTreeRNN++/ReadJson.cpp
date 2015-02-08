// Copyright (c) 2014 Piotr Mirowski. All rights reserved.
//                    piotr.mirowski@computer.org

#include <stdio.h>
#include <iostream>
#include <string>
#include "json.h"
#include "ReadJson.h"
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
  
  // Parse the JSON and keep a pointer to the root of the book
  json_value *_root = json_parse(source,
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
      for (json_value *token = u->first_child;
           token;
           token = token->next_sibling) {

        // Process the token to get:
        // its position in sentence,
        // word, discount and label
        string tokenWordAsContext(""), tokenWordAsTarget(""), tokenLabel("");
        double tokenDiscount = 0;
        int tokenPos = -1;
        ProcessToken(token, tokenPos, tokenWordAsTarget, tokenDiscount, tokenLabel);
        tokenWordAsContext = tokenWordAsTarget;

        // Concatenate word with label, when it is used as context?
        if (merge_label_with_word) {
          tokenWordAsContext += ":" + tokenLabel;
        }

        // Shall we insert new words/labels
        // into the vocabulary?
        if (insert_vocab) {
          if (merge_label_with_word) {
            if (tokenLabel == "LEAF") {
              // Insert target word to vocabulary
              corpus.InsertWord(tokenWordAsTarget, tokenDiscount);
            } else {
              // Insert concatenated context word and label to vocabulary
              corpus.InsertWord(tokenWordAsContext, tokenDiscount);
            }
          } else {
            // Insert word and label to two different vocabularies
            corpus.InsertWord(tokenWordAsContext, tokenDiscount);
            if (tokenLabel != "LEAF") {
              corpus.InsertLabel(tokenLabel);
            }
          }
        }
        // Insert new words to the book
        int wordIndexAsContext = 0, wordIndexAsTarget = 0, labelIndex = 0;
        if (merge_label_with_word) {
          wordIndexAsContext = corpus.LookUpWord(tokenWordAsContext);
          wordIndexAsTarget = corpus.LookUpWord(tokenWordAsTarget);
        } else {
          wordIndexAsContext = corpus.LookUpWord(tokenWordAsContext);
          wordIndexAsTarget = wordIndexAsContext;
          labelIndex = corpus.LookUpLabel(tokenLabel);
        }
        book->AddToken(isNewSentence, isNewUnroll,
                       tokenPos, wordIndexAsContext, wordIndexAsTarget,
                       tokenDiscount, labelIndex);
        // We are no longer at beginning of a sentence or unroll
        isNewSentence = false;
        isNewUnroll = false;
      }
      numUnrollsInThatSentence++;
    }
    numSentences++;
  }
  cout << "\nReadJSON: " << filename << endl;
  cout << "          (" << numSentences << " sentences, including empty ones; ";
  cout << book->NumTokens() << " tokens)\n";
  if (insert_vocab) {
    cout << "          Corpus now contains " << corpus.NumWords()
    << " words and " << corpus.NumLabels() << " labels\n";
  }
}


/**
 * Parse a token in the JSON parse tree to fill token data
 */
void ReadJson::ProcessToken(const json_value *_token,
                            int & tokenPos,
                            string & tokenWord,
                            double & tokenDiscount,
                            string & tokenLabel) const {
  // Safety check
  if (_token == NULL) {
    tokenPos = -1;
    tokenWord = "";
    tokenDiscount = 0;
    tokenLabel = "";
    return;
  }
  // Get the position of the token in the unroll
  json_value *element = _token->first_child;
  tokenPos = element->int_value;
  // Get the string of the word in the token in the unroll
  element = element->next_sibling;
  tokenWord = string(element->string_value);
  // Get the discount of the token in the unroll
  element = element->next_sibling;
  tokenDiscount = (double) 1.0 / (element->int_value);
  // Get the label of the token in the unroll
  element = element->next_sibling;
  tokenLabel = string(element->string_value);
}
