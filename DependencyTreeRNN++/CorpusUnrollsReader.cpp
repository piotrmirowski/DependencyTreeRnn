// Copyright (c) 2014-2015 Piotr Mirowski
//
// Piotr Mirowski, Andreas Vlachos
// "Dependency Recurrent Neural Language Models for Sentence Completion"
// ACL 2015

#include <stdio.h>
#include <climits>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include "CorpusUnrollsReader.h"
#include "ReadJson.h"

using namespace std;

/**
 * Add a token to the book
 */
void BookUnrolls::AddToken(bool isNewSentence, bool isNewUnroll,
                           int pos, int wordAsContext, int wordAsTarget,
                           double discount, int label) {
  
  // Add a new sentence?
  if (isNewSentence) {
    Sentence s;
    _sentences.push_back(s);
    _numUnrollsInSentence.push_back(0);
    vector<int> v;
    _numTokensInUnrollSentence.push_back(v);
    // Bookkeeping of sentences and unrolls
    _numSentences++;
    _sentenceIndex = _numSentences - 1;
    _unrollIndex = 0;
    _tokenIndex = 0;
  }
  // Add a new unroll?
  if (isNewUnroll) {
    Unroll u;
    _sentences[_sentenceIndex].push_back(u);
    // Bookkeeping of unrolls
    _numUnrollsInSentence[_sentenceIndex]++;
    _unrollIndex = _numUnrollsInSentence[_sentenceIndex] - 1;
    _numTokensInUnrollSentence[_sentenceIndex].push_back(0);
    _tokenIndex = 0;
  }
  // Add a new token
  Token newToken;
  newToken.pos = pos;
  newToken.wordAsContext = wordAsContext;
  newToken.wordAsTarget = wordAsTarget;
  newToken.discount = discount;
  newToken.label = label;
  _sentences[_sentenceIndex][_unrollIndex].push_back(newToken);
  _numTokensInUnrollSentence[_sentenceIndex][_unrollIndex]++;
  _numTokens++;
}


/**
 * Go to a specific sentence
 */
bool BookUnrolls::GoToSentence(int n) {
  // Sanity check
  if ((n < 0) || (n >= _numSentences)) {
    return false;
  }
  // Set the new sentence
  _sentenceIndex = n;
  // Reset the index of the unroll
  ResetUnroll();
  return true;
}


/**
 * Go to the next sentence
 */
int BookUnrolls::NextSentence() {
  // Set the new sentence by incrementing its index
  if (_sentenceIndex >= (_numSentences - 1)) {
    // Return to sentence 0
    ResetSentence();
  } else {
    // ... or simply go to the next sentence?
    _sentenceIndex++;
    // Reset the index of the unroll
    ResetUnroll();
  }
  return _sentenceIndex;
}


/**
 * Go to the next unroll in the sentence
 */
int BookUnrolls::NextUnrollInSentence() {
  int n_unrolls = _numUnrollsInSentence[_sentenceIndex];
  if (_unrollIndex >= (n_unrolls - 1)) {
    // Return to unroll 0 in the current sentence...
    ResetUnroll();
  } else {
    // ... or simply go to the next unroll?
    _unrollIndex++;
    // Reset the token in that unroll
    ResetToken();
  }
  return _unrollIndex;
}


/**
 * Go to the next unroll in the current sentence.
 * Here, we do not loop over but stop (return -1)
 * when the end of the unroll is reached.
 */
int BookUnrolls::NextTokenInUnroll() {
  // If we have reached the end of sentence
  if (_tokenIndex < 0)
    return -1;
  // Number of tokens in sentence
  int _numTokens = _numTokensInUnrollSentence[_sentenceIndex][_unrollIndex];
  // Go to the next token or stop
  if (_tokenIndex < (_numTokens - 1)) {
    _tokenIndex++;
    UpdateCurrentToken();
  } else {
    _tokenIndex = -1;
  }
  return _tokenIndex;
}


/**
 * Custom comparator for sorting a vector<pair<string, double>>
 * by values
 */
struct reverseSortByValue {
  bool operator() (const pair<string, double> &left,
                   const pair<string, double> &right) {
    return (left.second > right.second);
  }
};


/**
 * Filter and sort the vocabulary from another corpus
 */
void CorpusUnrolls::FilterSortVocabulary(CorpusUnrolls &other) {
  
  // Copy the labels as they are
  for (int k = 0; k < other.NumLabels(); k++) {
    InsertLabel(other.labelsReverse[k]);
  }
  
  // Initialize a vector of filtered word counts
  // that contains OOV and EOS
  vector<pair<string, double> > filteredWords;
  filteredWords.push_back(pair<string, double>("</s>", 0.0));
  filteredWords.push_back(pair<string, double>("<unk>", 0.0));
  double freqOOV = 0.0;
  double countWords = 0;
  
  // Copy only words with 3 or more occurrences into that vector
  // and keep statistics about OOV words.
  // Note that we start the indexing at 3 because we already stored
  // <unk> and </s>
  for (int k = 2; k < other.NumWords(); k++) {
    string word = other.vocabularyReverse[k];
    double wordFreq = ceil(other.wordCountsDiscounted[k]);
    if (wordFreq >= _minWordOccurrence) {
      pair<string, double> p(word, wordFreq);
      filteredWords.push_back(p);
    } else {
      freqOOV += wordFreq;
    }
    countWords += wordFreq;
  }
  // Set the number of </s> tokens to a large value
  filteredWords[0].second = INT_MAX;
  // Count the number of <unk>
  filteredWords[1].second = freqOOV;
  
  // Sort that vector by value
  // The sorting should keep </s> at position 0
  sort(filteredWords.begin(),
       filteredWords.end(),
       reverseSortByValue());
  
  // Completely clear the corpus word vocabulary
  // (not the labels)
  vocabulary.clear();
  vocabularyReverse.clear();
  wordCountsDiscounted.clear();
  _vocabSizeWords = 0;

  // Now we can set the number of </s> tokens to 0
  // (it never happens, because of the tree parsing)
  filteredWords[0].second = 0.0;
  
  // Copy the content of that vector
  for (int k = 0; k < filteredWords.size(); k++) {
    string word = filteredWords[k].first;
    double wordFreq = filteredWords[k].second;
    InsertWord(word, wordFreq);
  }
  // Note the OOV tag
  _oov = vocabulary["<unk>"];
}


/**
 * Copy the vocabulary from another corpus
 */
void CorpusUnrolls::CopyVocabulary(CorpusUnrolls &other) {
  
  // Completely clear the corpus word vocabulary and labels
  labels.clear();
  labelsReverse.clear();
  vocabulary.clear();
  vocabularyReverse.clear();
  wordCountsDiscounted.clear();
  _vocabSizeWords = 0;
  _vocabSizeLabels = 0;

  // Copy the labels as they are
  for (int k = 0; k < other.NumLabels(); k++) {
    InsertLabel(other.labelsReverse[k]);
  }
  
  // Insert the words from the other corpus into the vocabulary
  for (int k = 0; k < other.NumWords(); k++) {
    string word = other.vocabularyReverse[k];
    double wordFreq = other.wordCountsDiscounted[k];
    InsertWord(word, wordFreq);
  }

  // Note the OOV tag
  _oov = vocabulary["<unk>"];
}


/**
 * Export the vocabulary to a text file
 */
void CorpusUnrolls::ExportVocabulary(const string &filename) {
  // Write the header
  ofstream vocabFile(filename);
  vocabFile << NumWords() << "\t" << NumLabels() << "\n";
  // Write the labels
  for (int k = 0; k < NumLabels(); k++) {
    vocabFile << k << "\t" << labelsReverse[k] << "\n";
  }
  // Write the words and their discount factors
  for (int k = 0; k < NumWords(); k++) {
    vocabFile << k << "\t" << vocabularyReverse[k]
    << "\t" << wordCountsDiscounted[k] << "\n";
  }
  vocabFile.close();
}


/**
 * Import the vocabulary from a text file
 */
void CorpusUnrolls::ImportVocabulary(const string &filename) {

  // Read the header
  ifstream vocabFile(filename);
  cout << "Reading vocabulary file " << filename << endl;
  assert(vocabFile.is_open());

  // Completely clear the corpus word vocabulary and labels
  labels.clear();
  labelsReverse.clear();
  vocabulary.clear();
  vocabularyReverse.clear();
  wordCountsDiscounted.clear();
  _vocabSizeWords = 0;
  _vocabSizeLabels = 0;

  // Read the header line
  string line;
  getline(vocabFile, line);
  stringstream lineStream(line);
  string strNumWords;
  string strNumLabels;
  getline(lineStream, strNumWords, '\t');
  getline(lineStream, strNumLabels);
  int numWords = stoi(strNumWords);
  int numLabels = stoi(strNumLabels);
  cout << "Vocabulary file contains " << numWords << " words and "
  << numLabels << " labels\n";

  // Read the labels one by one
  for (int k = 0; k < numLabels; k++) {
    getline(vocabFile, line);
    stringstream lineStream(line);
    string strIdx;
    string label;
    getline(lineStream, strIdx, '\t');
    getline(lineStream, label);
    InsertLabel(label);
  }

  // Read the words one by one
  for (int k = 0; k < numWords; k++) {
    getline(vocabFile, line);
    stringstream lineStream(line);
    string strIdx;
    string word;
    string strWordFreq;
    getline(lineStream, strIdx, '\t');
    getline(lineStream, word, '\t');
    getline(lineStream, strWordFreq);
    double wordFreq = stof(strWordFreq);
    InsertWord(word, wordFreq);
  }

  vocabFile.close();

  // Note the OOV tag
  _oov = vocabulary["<unk>"];

  printf("Vocab size: %d\n", NumWords());
  printf("Unknown tag at: %d\n", _oov);
  printf("Label vocab size: %d\n", NumLabels());
}


/**
 * Read vocabulary from all books and return the number of tokens
 */
long CorpusUnrolls::ReadVocabulary(bool mergeLabel) {
  
  long nTokens = 0;
  // Loop over the books
  for (int k = 0; k < NumBooks(); k++) {
    // Open the training file, load it to a JSON structure
    // and add words to the corpus
    ReadJson *train_json =
    new ReadJson(_bookFilenames[k], *this, true, false, mergeLabel);
    nTokens = m_currentBook.NumTokens();
    // Free the memory
    delete train_json;
  }
  return nTokens;
}


/**
 * Read the current book into memory
 */
void CorpusUnrolls::ReadBook(bool mergeLabel) {
  
  // "Burn" the previous book, if any, to initialize it
  m_currentBook.Burn();
  // Open the training file, load it to a JSON structure
  // and add words to the corpus
  ReadJson *train_json =
  new ReadJson(_bookFilenames[_currentBookIndex], *this, false, true, mergeLabel);
  // Free the memory
  delete train_json;
}


/**
 * Insert a word into the vocabulary, if new
 */
int CorpusUnrolls::InsertWord(const string &word, double discount) {
  
  // Try to find the word
  int wordIndex = LookUpWord(word);
  if (wordIndex == _oov) {
    // Could not find word: insert it to the vocabulary
    wordIndex = _vocabSizeWords;
    pair<string, int> kv(word, wordIndex);
    vocabulary.insert(kv);
    pair<int, string> kv2(wordIndex, word);
    vocabularyReverse.insert(kv2);
    _vocabSizeWords++;
  } else {
    wordIndex = vocabulary[word];
  }
  
  // Find the current (dis)count of the word
  unordered_map<int, double>::iterator it2 =
  wordCountsDiscounted.find(wordIndex);
  if (it2 == wordCountsDiscounted.end()) {
    pair<int, double> kv(wordIndex, discount);
    wordCountsDiscounted.insert(kv);
  } else {
    wordCountsDiscounted[wordIndex] += discount;
  }
  
  // Simply return the word index
  return wordIndex;
}


/**
 * Insert a label into the vocabulary, if new
 */
int CorpusUnrolls::InsertLabel(const string &label) {
  
  // Try to find the label
  int labelIndex = LookUpLabel(label);
  if (labelIndex == -1) {
    // Could not find word: insert it to the vocabulary
    labelIndex = _vocabSizeLabels;
    pair<string, int> kv(label, labelIndex);
    labels.insert(kv);
    pair<int, string> kv2(labelIndex, label);
    labelsReverse.insert(kv2);
    _vocabSizeLabels++;
  } else {
    labelIndex = labels[label];
  }
  
  // Simply return the label index
  return labelIndex;
}


/**
 * Look-up a word in the vocabulary
 */
int CorpusUnrolls::LookUpWord(const string &word) {
  
  // Try to find the word
  int wordIndex = _oov;
  unordered_map<string, int>::iterator it =
  vocabulary.find(word);
  if (it != vocabulary.end()) {
    wordIndex = vocabulary[word];
  }
  return wordIndex;
}


/**
 * Look-up a label in the vocabulary
 */
int CorpusUnrolls::LookUpLabel(const string &label) {
  
  // Try to find the word
  int labelIndex = -1;
  unordered_map<string, int>::iterator it =
  labels.find(label);
  if (it != labels.end()) {
    labelIndex = labels[label];
  }
  return labelIndex;
}
