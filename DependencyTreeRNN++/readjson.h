// Copyright (c) 2014 Piotr Mirowski. All rights reserved.
//                    piotr.mirowski@computer.org

#ifndef DependencyTreeRNN___readjson_h
// Copyright (c) 2014 Piotr Mirowski. All rights reserved.
//                    piotr.mirowski@computer.org

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
  
  /**
   * Parse a corpus in the JSON parse tree and optionally print it
   */
  void TraverseCorpus(bool verbose);
  
  /**
   * Return the number of sentences
   */
  int NumSentences() { return (int)(_numUnrollsPerSentence.size()); }
  
  /**
   * Return the number of unrolls in a given sentence
   */
  int NumUnrolls(int n) { return _numUnrollsPerSentence[n]; }
  
  /**
   * Return the index of the current sentence
   */
  int CurrentSentenceIndex() { return _sentenceIndex; }
  
  /**
   * Return the index of the current sentence
   */
  int CurrentUnrollIndex() { return _unrollIndex; }
  
  /**
   * Go to a specific sentence
   */
  bool GoToSentence(int n);
  
  /**
   * Go to the next sentence
   */
  int NextSentence();
  
  /**
   * Go to the next unroll in the sentence
   */
  int NextUnrollInSentence();
  
  /**
   * Accessors to current token information
   */
  int CurrentTokenNumberInSentence() { return _tokenPos; }
  double CurrentTokenDiscount() { return _tokenDiscount; }
  char* CurrentTokenWord() { return _tokenWord; }
  char* CurrentTokenLabel() { return _tokenLabel; }
  
  /**
   * Go to the next unroll in the current sentence.
   * Here, we do not loop over but stop (return -1)
   * when the end of the unroll is reached.
   */
  int NextTokenInUnroll();
  
  /**
   * Reset the sentence in the current sentence
   */
  void ResetSentence();
  
  /**
   * Reset the unroll in the current sentence
   */
  void ResetUnroll();
  
  /**
   * Reset the token in the current sentence and unroll
   */
  void ResetToken();
  
protected:
  
  /**
   * JSON parse object for the book
   */
  json_value *_root;
  
  /**
   * JSON parse object for the current sentence,
   * curent unroll in the current sentence
   * and current token in the current unroll in the current sentence
   */
  json_value *_sentence;
  json_value *_unroll;
  json_value *_token;
  
  /**
   * Current sentence, unroll and token index
   */
  int _sentenceIndex;
  int _unrollIndex;
  int _tokenIndex;
  
  /**
   * Current token data
   */
  int _tokenPos;
  char* _tokenWord;
  double _tokenDiscount;
  char* _tokenLabel;
  
  /**
   * Number of unrolls per sentence
   */
  vector<int> _numUnrollsPerSentence;
  
  /**
   * Parse a corpus in the JSON parse tree and count sentences and unrolls
   */
  void CountSentencesUnrolls();
  
  /**
   * Parse a token in the JSON parse tree to fill token data
   */
  void ProcessToken();
  
  /**
   * Parse a token in the JSON parse tree to check the structure
   */
  void CheckToken();
};

#endif
