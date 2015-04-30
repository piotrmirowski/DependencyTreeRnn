// Copyright (c) 2014 Anonymized. All rights reserved.
//
// Code submitted as supplementary material for manuscript:
// "Dependency Recurrent Neural Language Models for Sentence Completion"
// Do not redistribute.

// Based on code by Geoffrey Zweig and Tomas Mikolov
// for the Recurrent Neural Networks Language Model (RNNLM) toolbox

#ifndef DependencyTreeRNN_RnnWeights_h
#define DependencyTreeRNN_RnnWeights_h

#include <stdio.h>
#include <vector>
#include <sstream>
#include "Utils.h"


//#define USE_HASHTABLES


/**
 * Tomas Mikolov decided to implement hash tables and hash functions
 * from scratch...
 */
const unsigned int c_Primes[] = {108641969, 116049371, 125925907, 133333309, 145678979, 175308587, 197530793, 234567803, 251851741, 264197411, 330864029, 399999781,
  407407183, 459258997, 479012069, 545678687, 560493491, 607407037, 629629243, 656789717, 716048933, 718518067, 725925469, 733332871, 753085943, 755555077,
  782715551, 790122953, 812345159, 814814293, 893826581, 923456189, 940740127, 953085797, 985184539, 990122807};
const unsigned int c_PrimesSize = sizeof(c_Primes)/sizeof(c_Primes[0]);


#ifdef USE_HASHTABLES

/**
 * Triple of integers that can be used as key in a hashtable
 */
struct WordTripleKey {
  int w1;
  int w2;
  int w3;

  WordTripleKey(int v1, int v2, int v3)
  : w1(v1), w2(v2), w3(v3) { }

  bool isValid() { return (w1 != -1) && (w2 != -1) && (w3 != -1); }

  bool operator==(const WordTripleKey &key) const {
    return ((w1 == key.w1) && (w2 == key.w2) && (w3 == key.w3));
  }
};


/**
 * Hashtable indexed by a triple of integers
 */
template <>
struct std::hash<WordTripleKey> {
  unsigned long long operator()(const WordTripleKey& k) const {
    unsigned long long hash = c_Primes[0] * c_Primes[1] * k.w1;
    hash += c_Primes[(2*c_Primes[1]+1)%c_PrimesSize] * k.w2;
    hash += c_Primes[(2*c_Primes[2]+2)%c_PrimesSize] * k.w3;
    return hash;
  }
};


/**
 * Pair of integers that can be used as key in a hashtable
 */
struct WordPairKey {
  int w1;
  int w2;

  WordPairKey(int v1, int v2)
  : w1(v1), w2(v2) { }

  bool isValid() { return (w1 != -1) && (w2 != -1); }

  bool operator==(const WordPairKey &key) const {
    return ((w1 == key.w1) && (w2 == key.w2));
  }
};


/**
 * Hashtable indexed by a pair of integers
 */
template <>
struct std::hash<WordPairKey>
{
  unsigned long long operator()(const WordPairKey& k) const {
    unsigned long long hash = c_Primes[0] * c_Primes[1] * k.w1;
    hash += c_Primes[(2*c_Primes[1]+1)%c_PrimesSize] * k.w2;
    return hash;
  }
};

#endif // USE_HASHTABLES

/**
 * Weights of an RNN
 */
class RnnWeights {
public:

  /**
   * Constructor
   */
  RnnWeights(int sizeVocabulary,
             int sizeHidden,
             int sizeFeature,
             int sizeClasses,
             int sizeCompress,
             long long sizeDirectConnection);

  /**
   * Load the weights matrices from a file
   */
  void Load(FILE *fi);

  /**
   * Clear the weights, before loading a new model, to save on memory
   */
  void Clear();

  /**
   * Save the weights matrices to a file
   */
  void Save(FILE *fo);

  // Weights between input and hidden layer
  std::vector<double> Input2Hidden;
  // Weights between former hidden state and current hidden layer
  std::vector<double> Recurrent2Hidden;
  // weights between features and hidden layer
  std::vector<double> Features2Hidden;
  // Weights between features and output layer
  std::vector<double> Features2Output;
  // Weights between hidden and output layer (or hidden and compression if compression>0)
  std::vector<double> Hidden2Output;
  // Optional weights between compression and output layer
  std::vector<double> Compress2Output;
  // Direct parameters between input and output layer
  // (similar to Maximum Entropy model parameters)
#ifdef USE_HASHTABLES
  std::unordered_map<WordTripleKey, float> DirectTriGram;
  std::unordered_map<WordPairKey, float> DirectBiGram;
  std::unordered_map<int, float> DirectUniGram;
#else
  std::vector<double> DirectNGram;
#endif

  /**
   * Return the number of direct connections between input words
   * and the output word (i.e., n-gram features)
   */
  int GetNumDirectConnection() const {
#ifdef USE_HASHTABLES
    return 1;
#else
    return static_cast<int>(DirectNGram.size());
#endif
  } // int GetNumDirectConnections()

  /**
   * Return the number of word classes
   */
  int GetNumClasses() const { return m_sizeClasses; }

  /**
   * Debug function
   */
  void Debug();

  
protected:

  /**
   * Dimensions of the network
   */
  int m_sizeVocabulary;
  int m_sizeHidden;
  int m_sizeFeature;
  int m_sizeClasses;
  int m_sizeCompress;
  long long m_sizeDirectConnection;
  int m_sizeInput;
  int m_sizeOutput;
}; // class RnnWeights

#endif
