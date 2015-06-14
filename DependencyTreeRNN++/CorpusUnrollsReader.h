// Copyright (c) 2014-2015 Piotr Mirowski
//
// Piotr Mirowski, Andreas Vlachos
// "Dependency Recurrent Neural Language Models for Sentence Completion"
// ACL 2015

#ifndef __DependencyTreeRNN____corpus__
#define __DependencyTreeRNN____corpus__

#include <stdlib.h>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <random>

/**
 * Basic unit of a text: a token
 */
struct Token {
  int pos;
  int wordAsContext;
  int wordAsTarget;
  double discount;
  int label;
};

/**
 * Sentence unroll: a vector of tokens
 */
typedef std::vector<Token> Unroll;

/**
 * Sentence: a vector of unrolls
 */
typedef std::vector<Unroll> Sentence;


/**
 * Book: a class containing a vector of sentences
 */
class BookUnrolls {
public:

  /**
   * Constructor and destructor
   */
  BookUnrolls() { Burn(); }
  ~BookUnrolls() { }

  /**
   * Wipe-out all content of the book
   */
  void Burn() {
    _sentences.clear();
    _numUnrollsInSentence.clear();
    _numTokensInUnrollSentence.clear();
    _numSentences = 0;
    _sentenceIndex = 0;
    _unrollIndex = 0;
    _tokenIndex = 0;
    _numTokens = 0;
  }

  /**
   * Add a token to the book
   */
  void AddToken(bool new_sentence, bool new_unroll,
                int pos, int wordAsContext, int wordAsTarget,
                double discount, int label);

  /**
   * Return the number of sentences
   */
  int NumSentences() { return _numSentences; }

  /**
   * Return the number of unrolls in sentence
   */
  int NumUnrolls(int k) { return _numUnrollsInSentence[k]; }

  /**
   * Return the number of tokens in unroll of a sentence
   */
  int NumTokens(int k, int j) { return _numTokensInUnrollSentence[k][j]; }

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
   * Go to the next unroll in the current sentence.
   * Here, we do not loop over but stop (return -1)
   * when the end of the unroll is reached.
   */
  int NextTokenInUnroll();

  /**
   * Update the current token
   */
  void UpdateCurrentToken() {
    _currentToken =
    &(_sentences[_sentenceIndex][_unrollIndex][_tokenIndex]);
  }

  /**
   * Accessors to the current token's information
   */
  int CurrentTokenNumberInSentence() { return _currentToken->pos; }
  double CurrentTokenDiscount() { return _currentToken->discount; }
  int CurrentTokenWordAsContext() { return _currentToken->wordAsContext; }
  int CurrentTokenWordAsTarget() { return _currentToken->wordAsTarget; }
  int CurrentTokenLabel() { return _currentToken->label; }

  /**
   * Reset the sentence in the current sentence
   */
  void ResetSentence() {
    _sentenceIndex = 0;
    // Recursively reset the unroll of that first sentence
    ResetUnroll();
  }

  /**
   * Reset the unroll in the current sentence
   */
  void ResetUnroll() {
    _unrollIndex = 0;
    // Recursively reset the token of that first unroll
    ResetToken();
  }

  /**
   * Reset the token in the current sentence and unroll
   */
  void ResetToken() {
    _tokenIndex = 0;
    UpdateCurrentToken();
  }

  /**
   * Number of tokens
   */
  long NumTokens() { return _numTokens; }

protected:

  // All the sentences of the book
  std::vector<Sentence> _sentences;

  // Copy of the current token
  Token *_currentToken;

  // Current sentence, unroll and token index
  int _sentenceIndex;
  int _unrollIndex;
  int _tokenIndex;

  // Number of sentences
  int _numSentences;

  // Number of unrolls in each sentence
  std::vector<int> _numUnrollsInSentence;

  // Number of tokens in each unroll and sentence
  std::vector<std::vector<int> > _numTokensInUnrollSentence;

  // Total number of tokens
  long _numTokens;
};


/**
 * CorpusUnrolls: contains all vocabulary and the list of books
 * but stores only one book at a time
 */
class CorpusUnrolls {
public:
  /**
   * Constructor
   */
  CorpusUnrolls() :
  _minWordOccurrence(3),
  _oov(0),
  _vocabSizeWords(0),
  _vocabSizeLabels(0),
  _currentBookIndex(-1) {
    // Insert OOV and EOS tokens
    InsertWord("<unk>", 1.0);
    InsertWord("</s>", 1.0);
    // Insert ROOT label
    InsertLabel("ROOT");
  }

  /**
   * Constructor and destructor
   */
  ~CorpusUnrolls () { }

public:
  /**
   * Number of books
   */
  int NumBooks() { return (int)(_bookFilenames.size()); }

  /**
   * Size of the vocabulary
   */
  int NumWords() { return _vocabSizeWords; }

  /**
   * Number of labels
   */
  int NumLabels() { return _vocabSizeLabels; }

  /**
   * Look-up a word in the vocabulary
   */
  int LookUpWord(const std::string &word);

  /**
   * Look-up a label in the vocabulary
   */
  int LookUpLabel(const std::string &label);

public:
  /**
   * Set minimum number of word occurrences
   */
  void SetMinWordOccurrence(int val) { _minWordOccurrence = val; }

  /**
   * Insert a word into the vocabulary, if new
   */
  int InsertWord(const std::string &word, double discount);

  /**
   * Insert a label into the vocabulary, if new
   */
  int InsertLabel(const std::string &label);

  /**
   * Read vocabulary from all books and return the number of tokens
   */
  long ReadVocabulary(bool mergeLabel);

  /**
   * Filter and sort the vocabulary from another corpus
   */
  void FilterSortVocabulary(CorpusUnrolls &other);

  /**
   * Copy the vocabulary from another corpus
   */
  void CopyVocabulary(CorpusUnrolls &other);

  /**
   * Export the vocabulary to a text file
   */
  void ExportVocabulary(const std::string &filename);

  /**
   * Import the vocabulary from a text file
   */
  void ImportVocabulary(const std::string &filename);

  /**
   * Add a book
   */
  void AddBookFilename(const std::string &filename) {
    _bookFilenames.push_back(filename);
    NextBook();
  }

  /**
   * Go to next book
   */
  int NextBook() {
    _currentBookIndex++;
    if (_currentBookIndex == NumBooks()) { _currentBookIndex = 0; }
    return _currentBookIndex;
  }

  /**
   * Shuffle the order of the books
   */
  void ShuffleBooks() {
    std::random_shuffle(_bookFilenames.begin(), _bookFilenames.end());
  }

  /**
   * Read the current book into memory
   */
  void ReadBook(bool mergeLabel);

protected:

  // Minimum number of word occurrences not to be OOV
  int _minWordOccurrence;

  // Out-of-vocabulary token
  int _oov;

  // Number of words and labels in the vocabulary
  int _vocabSizeWords;
  int _vocabSizeLabels;

  // Current book
  int _currentBookIndex;

  // List of books (filenames)
  std::vector<std::string> _bookFilenames;

public:

  // Vocabulary: map between a string of text and an integer
  std::unordered_map<std::string, int> vocabulary;
  std::unordered_map<int, std::string> vocabularyReverse;

  // Discounted word counts
  std::unordered_map<int, double> wordCountsDiscounted;

  // Labels: map between a string of text and an integer
  std::unordered_map<std::string, int> labels;
  std::unordered_map<int, std::string> labelsReverse;

  // Current book
  BookUnrolls m_currentBook;
};

#endif /* defined(__DependencyTreeRNN____corpus__) */
