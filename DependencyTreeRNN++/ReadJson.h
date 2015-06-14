// Copyright (c) 2014-2015 Piotr Mirowski
//
// Piotr Mirowski, Andreas Vlachos
// "Dependency Recurrent Neural Language Models for Sentence Completion"
// ACL 2015

#ifndef DependencyTreeRNN___readjson_h
#define DependencyTreeRNN___readjson_h

#include <vector>
#include "CorpusUnrollsReader.h"

using namespace std;

struct JsonToken {
  int pos;
  string word;
  double discount;
  string label;
};


class ReadJson {
public:
  
  /**
   * Constructor: read a text file in JSON format.
   * If required, insert words and labels to the vocabulary.
   * If required, insert tokens into the current book.
   */
  ReadJson(const string &filename,
           CorpusUnrolls &corpus,
           bool insert_vocab,
           bool read_book,
           bool merge_label_with_word);
  
  /**
   * Destructor
   */
  ~ReadJson() { }

protected:
  

  /**
   * Trim a word
   */
  string const Trim(const string &word) const;

  /**
   * Parse a token
   */
  size_t const ParseToken(const string &json_element,
                          JsonToken &tok) const;

  /**
   * Parse an unroll
   */
  size_t const ParseUnroll(const string &json_unrolls,
                           vector<JsonToken> &unroll) const;

  /**
   * Parse a sentence
   */
  size_t const ParseSentence(const string &json_sentences,
                             vector<vector<JsonToken>> &sentence) const;

  /**
   * Parse a book
   */
  size_t const ParseBook(const string &json_book,
                         vector<vector<vector<JsonToken>>> &book) const;
};

#endif
