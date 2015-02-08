// Copyright (c) 2014 Piotr Mirowski. All rights reserved.
//                    piotr.mirowski@computer.org

#ifndef DependencyTreeRNN___readjson_h
#define DependencyTreeRNN___readjson_h

#include <vector>
#include "json.h"
#include "CorpusUnrollsReader.h"

using namespace std;


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
   * Parse a token in the JSON parse tree to fill token data
   */
  void ProcessToken(const json_value *_token,
                    int & tokenPos, string & tokenWord,
                    double & tokenDiscount, string & tokenLabel) const;
};

#endif
