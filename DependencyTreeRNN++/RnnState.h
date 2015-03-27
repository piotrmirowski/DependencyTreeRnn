// Copyright (c) 2014 Piotr Mirowski. All rights reserved.
//                    piotr.mirowski@computer.org
//
// Based on code by Geoffrey Zweig and Tomas Mikolov
// for the Recurrent Neural Networks Language Model (RNNLM) toolbox
//
// Recurrent neural network based statistical language modeling toolkitsize
// Version 0.3f
// (c) 2010-2012 Tomas Mikolov (tmikolov@gmail.com)
// Extensions from 0.3e to 0.3f version done at Microsoft Research
//
// This code implements the following paper:
//   Tomas Mikolov and Geoffrey Zweig
//   "Context Dependent Recurrent Neural Network Language Model"
//   Microsoft Research Technical Report MSR-TR-2012-92 July 27th, 2012
//   IEEE Conference on Spoken Language Technologies
//   http://research.microsoft.com/apps/pubs/default.aspx?id=176926
//
// Contributions: Piotr Mirowski (piotr.mirowski@computer.org)
//                Geoffrey Zweig (gzweig@microsoft.com)
//                Davide Di Gennaro (dadigenn@microsoft.com)
//                Vitaly Vazhnais (v-vitalv@microsoft.com)
//                Francesco Nidito (frnidito@microsoft.com)
//                Daniel Voinea (frnidito@microsoft.com)

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

#ifndef DependencyTreeRNN___RnnState_h
#define DependencyTreeRNN___RnnState_h

#include <vector>
#include <algorithm>


/**
 * Max n-gram order, used for word history and direct connections
 * from the word history to the word output
 */
const int c_maxNGramOrder = 20;


/**
 * State vectors in the RNN model, storing per-word and per-class activations
 */
class RnnState {
public:

  /**
   * Constructor
   */
  RnnState(int sizeVocabulary,
           int sizeHidden,
           int sizeFeature,
           int sizeClasses,
           int sizeCompress,
           long long sizeDirectConnection,
           int orderDirectConnection)
  : m_orderDirectConnection(orderDirectConnection) {
    int sizeInput = sizeVocabulary;
    int sizeOutput = sizeVocabulary + sizeClasses;
    WordHistory.assign(c_maxNGramOrder, 0);
    InputLayer.assign(sizeInput, 0.0);
    InputGradient.assign(sizeInput, 0.0);
    RecurrentLayer.assign(sizeHidden, 0.0);
    RecurrentGradient.assign(sizeHidden, 0.0);
    HiddenLayer.assign(sizeHidden, 0.0);
    HiddenGradient.assign(sizeHidden, 0.0);
    FeatureLayer.assign(sizeFeature, 0.0);
    FeatureGradient.assign(sizeFeature, 0.0);
    OutputLayer.assign(sizeOutput, 0.0);
    OutputGradient.assign(sizeOutput, 0.0);
    CompressLayer.assign(sizeCompress, 0.0);
    CompressGradient.assign(sizeCompress, 0.0);
  }

  // Input layer (i.e., words)
  std::vector<double> InputLayer;
  // Input feature layer (e.g., topics)
  std::vector<double> FeatureLayer;
  // Hidden layer at previous time step
  std::vector<double> RecurrentLayer;
  // Hidden layer
  std::vector<double> HiddenLayer;
  // Second (compression) hidden layer
  std::vector<double> CompressLayer;
  // Output layer
  std::vector<double> OutputLayer;

  // Gradient to the words in input layer
  std::vector<double> InputGradient;
  // Gradient to the features in input layer
  std::vector<double> FeatureGradient;
  // Gradient to the hidden state at previous time step
  std::vector<double> RecurrentGradient;
  // Gradient to the hidden layer
  std::vector<double> HiddenGradient;
  // Gradient to the second (compression) hidden layer
  std::vector<double> CompressGradient;
  // Gradient to the output layer
  std::vector<double> OutputGradient;

  // Word history
  std::vector<int> WordHistory;


  /**
   * Return the number of units in the input (word) layer.
   */
  int GetInputSize() const {
    return static_cast<int>(InputLayer.size());
  }


  /**
   * Return the number of units in the input (word) layer.
   */
  int GetHiddenSize() const {
    return static_cast<int>(HiddenLayer.size());
  }


  /**
   * Return the number of units in the optional hidden compression layer.
   */
  int GetCompressSize() const {
    return static_cast<int>(CompressLayer.size());
  }


  /**
   * Return the number of units in the feature (e.g., topic) layer.
   */
  int GetFeatureSize() const {
    return static_cast<int>(FeatureLayer.size());
  }


  /**
   * Return the number of units in the output layer.
   */
  int GetOutputSize() const {
    return static_cast<int>(OutputLayer.size());
  }


  /**
   * Return the number of units in the output layer.
   */
  int GetOrderDirectConnection() const { return m_orderDirectConnection; }

protected:
  int m_orderDirectConnection;
};


class RnnBptt {
public:

  /**
   * Constructor
   */
  RnnBptt(int sizeVocabulary, int sizeHidden, int sizeFeature,
          int numBpttSteps, int bpttBlockSize)
  : m_bpttSteps(numBpttSteps), m_bpttBlock(bpttBlockSize),
  m_sizeHidden(sizeHidden), m_sizeFeature(sizeFeature),
  m_steps(0) {
    Reset();
    WeightsInput2Hidden.assign(sizeVocabulary * sizeHidden, 0);
    WeightsRecurrent2Hidden.assign(sizeHidden * sizeHidden, 0);
    WeightsFeature2Hidden.assign(sizeFeature * sizeHidden, 0);
  }


  /**
   * Number of BPTT steps that can be considered
   */
  int NumSteps() { return m_steps; }


  /**
   * Reset the BPTT memory
   */
  void Reset() {
    m_steps = 0;
    History.assign(m_bpttSteps + m_bpttBlock + 10, -1);
    HiddenLayer.assign((m_bpttSteps + m_bpttBlock + 1) * m_sizeHidden, 0);
    HiddenGradient.assign((m_bpttSteps + m_bpttBlock + 1) * m_sizeHidden, 0);
    FeatureLayer.assign((m_bpttSteps + m_bpttBlock + 2) * m_sizeFeature, 0);
    FeatureGradient.assign((m_bpttSteps + m_bpttBlock + 2) * m_sizeFeature, 0);
  }


  /**
   * Shift the BPTT memory by one
   */
  void Shift(int lastWord) {
    // Shift memory needed for BPTT to next time step
    if (m_bpttSteps > 0) {
      // shift memory needed for bptt to next time step
      for (int a = m_bpttSteps + m_bpttBlock - 1; a > 0; a--) {
        History[a] = History[a - 1];
      }
      History[0] = lastWord;

      for (int a = m_bpttSteps + m_bpttBlock - 1; a > 0; a--) {
        for (int b = 0; b < m_sizeHidden; b++) {
          HiddenLayer[a * m_sizeHidden + b] =
          HiddenLayer[(a - 1) * m_sizeHidden + b];
          HiddenGradient[a * m_sizeHidden + b] =
          HiddenGradient[(a - 1) * m_sizeHidden + b];
        }
      }

      for (int a = m_bpttSteps + m_bpttBlock - 1; a > 0; a--) {
        for (int b = 0; b < m_sizeFeature; b++) {
          FeatureLayer[a * m_sizeFeature+b] =
          FeatureLayer[(a - 1) * m_sizeFeature+b];
        }
      }
    }
    // Keep track of the number of that can be considered for BPTT
    m_steps++;
    m_steps = std::min(m_steps, m_bpttSteps + m_bpttBlock);
  }


  // Word history
  std::vector<int> History;
  // History of hidden layer inputs
  std::vector<double> HiddenLayer;
  // History of gradients to the hidden layer
  std::vector<double> HiddenGradient;
  // History of feature inputs
  std::vector<double> FeatureLayer;
  // History of gradients to the feature layer
  std::vector<double> FeatureGradient;
  // Gradients to the weights, to be added to the SGD gradients
  std::vector<double> WeightsInput2Hidden;
  std::vector<double> WeightsRecurrent2Hidden;
  std::vector<double> WeightsFeature2Hidden;


protected:
  // Number of steps gradients are back-propagated through time
  int m_bpttSteps;
  // How many steps (words) do we wait between consecutive BPTT?
  int m_bpttBlock;
  // How many steps have been stored since the last reset?
  int m_steps;
  // Number of hidden nodes
  int m_sizeHidden;
  // Number of features
  int m_sizeFeature;
};

#endif
