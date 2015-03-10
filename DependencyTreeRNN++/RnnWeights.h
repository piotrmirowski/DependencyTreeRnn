//
//  RnnWeights.h
//  DependencyTreeRNN
//
//  Created by Piotr Mirowski on 03/03/2015.
//  Copyright (c) 2015 Piotr Mirowski. All rights reserved.
//

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
             long long sizeDirectConnection)
  : m_sizeVocabulary(sizeVocabulary),
  m_sizeHidden(sizeHidden),
  m_sizeFeature(sizeFeature),
  m_sizeClasses(sizeClasses),
  m_sizeCompress(sizeCompress),
  m_sizeDirectConnection(sizeDirectConnection),
  m_sizeInput(sizeVocabulary),
  m_sizeOutput(sizeVocabulary + sizeClasses) {

    // Sanity check
    assert(sizeClasses <= sizeVocabulary);
    std::cout << "Allocate RnnWeights: " << m_sizeInput << " inputs ("
    << sizeVocabulary << " words), "
    << m_sizeClasses << " classes, "
    << m_sizeHidden << " hiddens, "
    << m_sizeFeature << " features, "
    << m_sizeCompress << " compressed\n"
    << m_sizeDirectConnection << "M direct connections\n";

    // Allocate the weights connecting those layers
    // (will be assigned random values later)
    Input2Hidden.resize(m_sizeInput * m_sizeHidden);
    Recurrent2Hidden.resize(m_sizeHidden * m_sizeHidden);
    Features2Hidden.resize(m_sizeFeature * m_sizeHidden);
    Features2Output.resize(m_sizeFeature * m_sizeOutput);
    if (sizeCompress == 0) {
      Hidden2Output.resize(m_sizeHidden * m_sizeOutput);
    } else {
      // Add a compression layer between hidden nodes and outputs
      Hidden2Output.resize(m_sizeHidden * m_sizeCompress);
      Compress2Output.resize(m_sizeCompress * m_sizeOutput);
    }
    // Change that to proper normal distribution
    // http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    RandomizeVector(Input2Hidden);
    RandomizeVector(Recurrent2Hidden);
    if (sizeFeature > 0) {
      RandomizeVector(Features2Hidden);
      RandomizeVector(Features2Output);
    }
    if (sizeCompress > 0) {
      RandomizeVector(Compress2Output);
    }
    RandomizeVector(Hidden2Output);

    // Initialize the direct n-gram connections
#ifdef USE_HASHTABLES
    DirectBiGram.clear();
    DirectTriGram.clear();
#else
    DirectNGram.assign(m_sizeDirectConnection, 0.0);
#endif
  } // RnnWeights()


  /**
   * Load the weights matrices from a file
   */
  void Load(FILE *fi) {
    // Read the weights of input -> hidden connections
    ReadBinaryMatrix(fi, m_sizeInput, m_sizeHidden, Input2Hidden);
    // Read the weights of recurrent hidden -> hidden connections
    ReadBinaryMatrix(fi, m_sizeHidden, m_sizeHidden, Recurrent2Hidden);
    // Read the weights of feature -> hidden connections
    ReadBinaryMatrix(fi, m_sizeFeature, m_sizeHidden, Features2Hidden);
    // Read the weights of feature -> output connections
    ReadBinaryMatrix(fi, m_sizeFeature, m_sizeOutput, Features2Output);
    if (m_sizeCompress == 0) {
      // Read the weights of hidden -> output connections
      ReadBinaryMatrix(fi, m_sizeHidden, m_sizeOutput, Hidden2Output);
    } else {
      // Read the weights of hidden -> compression connections
      ReadBinaryMatrix(fi, m_sizeHidden, m_sizeCompress, Hidden2Output);
      // Read the weights of compression -> output connections
      ReadBinaryMatrix(fi, m_sizeCompress, m_sizeOutput, Compress2Output);
    }
    if (m_sizeDirectConnection > 0) {
#ifdef USE_HASHTABLES
#else
      // Read the direct connections
      ReadBinaryVector(fi, m_sizeDirectConnection, DirectNGram);
#endif
    }
    Debug();
  } // void Load()


  /**
   * Save the weights matrices to a file
   */
  void Save(FILE *fo) {
    // Save the weights U: input -> hidden (i.e., the word embeddings)
    printf("Saving %dx%d input->hidden weights...\n", m_sizeHidden, m_sizeInput);
    SaveBinaryMatrix(fo, m_sizeInput, m_sizeHidden, Input2Hidden);
    // Save the weights W: recurrent hidden -> hidden (i.e., the time-delay)
    printf("Saving %dx%d recurrent hidden->hidden weights...\n",
           m_sizeHidden, m_sizeHidden);
    SaveBinaryMatrix(fo, m_sizeHidden, m_sizeHidden, Recurrent2Hidden);
    // Save the weights feature -> hidden
    printf("Saving %dx%d feature->hidden weights...\n", m_sizeHidden, m_sizeFeature);
    SaveBinaryMatrix(fo, m_sizeFeature, m_sizeHidden, Features2Hidden);
    // Save the weights G: feature -> output
    printf("Saving %dx%d feature->output weights...\n", m_sizeOutput, m_sizeFeature);
    SaveBinaryMatrix(fo, m_sizeFeature, m_sizeOutput, Features2Output);
    // Save the weights hidden -> compress and compress -> output
    // or simply the weights V: hidden -> output
    if (m_sizeCompress > 0) {
      printf("Saving %dx%d hidden->compress weights...\n", m_sizeCompress, m_sizeHidden);
      SaveBinaryMatrix(fo, m_sizeHidden, m_sizeCompress, Hidden2Output);
      printf("Saving %dx%d compress->output weights...\n", m_sizeCompress, m_sizeOutput);
      SaveBinaryMatrix(fo, m_sizeCompress, m_sizeOutput, Compress2Output);
    } else {
      printf("Saving %dx%d hidden->output weights...\n", m_sizeOutput, m_sizeHidden);
      SaveBinaryMatrix(fo, m_sizeHidden, m_sizeOutput, Hidden2Output);
    }
    if (m_sizeDirectConnection > 0) {
      // Save the direct connections
      printf("Saving %lld n-gram connections...\n", m_sizeDirectConnection);
#ifdef USE_HASHTABLES
#else
      for (long long aa = 0; aa < m_sizeDirectConnection; aa++) {
        float fl = (float)(DirectNGram[aa]);
        fwrite(&fl, 4, 1, fo);
      }
#endif
    }
    Debug();
  } // void Save()


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


  /**
   * Debug function
   */
  void Debug() {
    std::cout << "input2hidden: " << m_sizeInput << " " << m_sizeHidden << " "
    << Input2Hidden[100] << std::endl;
    std::cout << "recurrent2hidden: " << m_sizeHidden << " " << m_sizeHidden << " "
    << Recurrent2Hidden[100] << std::endl;
    std::cout << "hidden2output: " << m_sizeHidden << " " << m_sizeOutput << " "
    << Hidden2Output[100] << std::endl;
    std::cout << "features2hidden: " << m_sizeFeature << " " << m_sizeHidden << " "
    << Features2Hidden[100] << std::endl;
    std::cout << "features2output: " << m_sizeFeature << " " << m_sizeOutput << " "
    << Features2Output[100] << std::endl;
    if (m_sizeDirectConnection > 0)
      std::cout << "direct: " << m_sizeDirectConnection << " "
      << DirectNGram[100] << std::endl;
  } // void Debug()
}; // class RnnWeights


#endif
