// Copyright (c) 2014-2015 Piotr Mirowski
//
// Piotr Mirowski, Andreas Vlachos
// "Dependency Recurrent Neural Language Models for Sentence Completion"
// ACL 2015

// Based on code by Geoffrey Zweig and Tomas Mikolov
// for the Feature-Augmented RNN Tool Kit
// http://research.microsoft.com/en-us/projects/rnn/

/*
 This file is based on or incorporates material from the projects listed below (collectively, "Third Party Code").
 Microsoft is not the original author of the Third Party Code. The original copyright notice and the license under which Microsoft received such Third Party Code,
 are set forth below. Such licenses and notices are provided for informational purposes only. Microsoft, not the third party, licenses the Third Party Code to you
 under the terms set forth in the EULA for the Microsoft Product. Microsoft reserves all rights not expressly granted under this agreement, whether by implication,
 estoppel or otherwise.

 RNNLM 0.3e by Tomas Mikolov

 Provided for Informational Purposes Only

 BSD License
 All rights reserved.
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other
 materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef DependencyTreeRNN_RnnWeights_h
#define DependencyTreeRNN_RnnWeights_h

#include <stdio.h>
#include <vector>
#include <sstream>
#include "Utils.h"


/**
 * Tomas Mikolov decided to implement hash tables and hash functions
 * from scratch...
 */
const unsigned int c_Primes[] = {108641969, 116049371, 125925907, 133333309, 145678979, 175308587, 197530793, 234567803, 251851741, 264197411, 330864029, 399999781,
  407407183, 459258997, 479012069, 545678687, 560493491, 607407037, 629629243, 656789717, 716048933, 718518067, 725925469, 733332871, 753085943, 755555077,
  782715551, 790122953, 812345159, 814814293, 893826581, 923456189, 940740127, 953085797, 985184539, 990122807};
const unsigned int c_PrimesSize = sizeof(c_Primes)/sizeof(c_Primes[0]);


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
  std::vector<double> DirectNGram;

  /**
   * Return the number of direct connections between input words
   * and the output word (i.e., n-gram features)
   */
  int GetNumDirectConnection() const {
    return static_cast<int>(DirectNGram.size());
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
